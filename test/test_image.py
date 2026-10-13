#!/usr/bin/env python3
"""Tests for pyhik.image and the ISAPIClient image-settings methods across multiple camera fixtures."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from pyhik.image import (
    ImageSettingType,
    parse_image_capabilities,
    parse_image_values,
)
from pyhik.isapi import (
    ISAPIAuthError,
    ISAPIClient,
    ISAPIError,
    ISAPINotFoundError,
    RawResponse,
)
from test.common import (
    DEFAULT_HOST,
    DEFAULT_USERNAME,
    DEFAULT_PASSWORD,
    DEFAULT_HTTP_PORT,
    DEFAULT_HTTPS_PORT,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture_pairs():
    """Dynamically discover capability and value XML pairs in the fixtures folder."""
    pairs = []
    # Looks for files matching image_capabilities_*.xml
    for cap_path in FIXTURES.glob("image_capabilities_*.xml"):
        suffix = cap_path.name[len("image_capabilities_"):-len(".xml")]
        val_path = FIXTURES / f"image_values_{suffix}.xml"
        if val_path.exists():
            pairs.append((suffix, cap_path.read_bytes(), val_path.read_bytes()))

    # Fallback if only the legacy file exists without a suffix pattern
    if not pairs:
        legacy_cap = FIXTURES / "image_capabilities_fixed_basic.xml"
        legacy_val = FIXTURES / "image_values_fixed_basic.xml"
        if legacy_cap.exists() and legacy_val.exists():
            pairs.append(("legacy_basic", legacy_cap.read_bytes(), legacy_val.read_bytes()))

    return pairs


def by_path(settings):
    return {s.path: s for s in settings}


class TestParseImageCapabilitiesMultiModel(unittest.TestCase):
    """Parameterized parsing tests across all discovered camera model fixtures."""

    def setUp(self):
        self.fixture_pairs = load_fixture_pairs()
        if not self.fixture_pairs:
            self.fail("No image fixture XML files found in test/fixtures/")

    def test_fixture_parsing_across_models(self):
        """Run core structure and type validations against every camera fixture."""
        for model_name, caps, values in self.fixture_pairs:
            with self.subTest(model=model_name):
                settings = parse_image_capabilities(caps, values)
                s_map = by_path(settings)

                # Ensure settings were successfully parsed
                self.assertGreater(len(settings), 0)

                # Common base checks applicable across standard fixed/ptz models
                if "WDR/mode" in s_map:
                    self.assertEqual(s_map["WDR/mode"].type, ImageSettingType.SELECT)
                if "HLC/enabled" in s_map:
                    self.assertEqual(s_map["HLC/enabled"].type, ImageSettingType.SWITCH)


class TestParseImageCapabilitiesSpecific(unittest.TestCase):
    """Detailed parsing tests using the primary reference fixture."""

    @classmethod
    def setUpClass(cls):
        pairs = load_fixture_pairs()
        if not pairs:
            raise unittest.SkipError("No fixtures available")
        # Use the first available pair for strict assertions
        cls.model_name, cls.caps, cls.values = pairs[0]

    def setUp(self):
        self.settings = parse_image_capabilities(self.caps, self.values)
        self.by_path = by_path(self.settings)

    def test_types(self):
        s = self.by_path
        self.assertEqual(s["WDR/mode"].type, ImageSettingType.SELECT)
        self.assertEqual(s["WDR/WDRLevel"].type, ImageSettingType.NUMBER)
        self.assertEqual(s["HLC/enabled"].type, ImageSettingType.SWITCH)

    def test_select_options_are_raw_values(self):
        self.assertEqual(self.by_path["WDR/mode"].options, ["close", "open", "auto"])
        self.assertEqual(
            self.by_path["WhiteBalance/WhiteBalanceStyle"].options[:3],
            ["manual", "auto1", "locked"],
        )

    def test_number_range(self):
        level = self.by_path["Exposure/OverexposeSuppress/DistanceLevel"]
        self.assertEqual((level.min_value, level.max_value), (1.0, 100.0))

    def test_numeric_options_stay_a_select(self):
        level = self.by_path["IrcutFilter/nightToDayFilterLevel"]
        self.assertEqual(level.type, ImageSettingType.SELECT)
        self.assertEqual(level.options, [str(n) for n in range(8)])

    def test_current_values_come_from_values_document(self):
        self.assertEqual(self.by_path["Shutter/ShutterLevel"].value, "1/1000")
        self.assertEqual(self.by_path["Color/brightnessLevel"].value, "80")

    def test_value_falls_back_to_capabilities_default(self):
        advanced = self.by_path["NoiseReduce/AdvancedMode/FrameNoiseReduceLevel"]
        self.assertEqual(advanced.value, "19")

    def test_empty_default_is_none(self):
        self.assertIsNone(self.by_path["ImageFlip/ImageFlipStyle"].value)

    def test_structural_paths_excluded(self):
        for path in ("id", "enabled", "videoInputID"):
            self.assertNotIn(path, self.by_path)

    def test_elements_without_opt_or_range_are_not_settings(self):
        self.assertNotIn("IrcutFilter/Schedule/scheduleType", self.by_path)
        self.assertNotIn("isSupportLaserSpotManual", self.by_path)

    def test_single_option_select_is_kept(self):
        exposure = self.by_path["Exposure/ExposureType"]
        self.assertEqual(exposure.type, ImageSettingType.SELECT)
        self.assertEqual(exposure.options, ["manual"])

    def test_blc_enabled_and_mode_are_merged(self):
        mode = self.by_path["BLC/BLCMode"]
        self.assertNotIn("BLC/enabled", self.by_path)
        self.assertEqual(mode.linked_enabled_path, "BLC/enabled")
        self.assertEqual(mode.off_value, "CLOSE")
        self.assertEqual(mode.value, "AUTO")

    def test_unmerged_enabled_switches_remain(self):
        for path in ("HLC/enabled", "ImageFlip/enabled", "corridor/enabled"):
            self.assertEqual(self.by_path[path].type, ImageSettingType.SWITCH)

    def test_total_count(self):
        self.assertEqual(len(self.settings), 30)


class TestParseVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pairs = load_fixture_pairs()
        cls.caps = pairs[0][1] if pairs else b""
        cls.values = pairs[0][2] if pairs else b""

    def test_disabled_linked_select_reports_off_value(self):
        values = self.values.replace(
            b"<enabled>true</enabled>\n<BLCMode>AUTO</BLCMode>",
            b"<enabled>false</enabled>",
        )
        self.assertNotEqual(values, self.values)
        mode = by_path(parse_image_capabilities(self.caps, values))["BLC/BLCMode"]
        self.assertEqual(mode.value, "CLOSE")

    def test_without_values_document_uses_capabilities_defaults(self):
        settings = by_path(parse_image_capabilities(self.caps))
        self.assertEqual(settings["WDR/WDRLevel"].value, "41")
        self.assertEqual(settings["BLC/BLCMode"].value, "AUTO")

    def test_accepts_text_input(self):
        settings = parse_image_capabilities(self.caps.decode("utf-8"), self.values.decode("utf-8"))
        self.assertEqual(len(settings), 30)

    def test_namespace_agnostic(self):
        other = self.caps.replace(
            b"http://www.hikvision.com/ver20/XMLSchema",
            b"http://www.isapi.org/ver20/XMLSchema",
        )
        self.assertEqual(
            by_path(parse_image_capabilities(other)).keys(),
            by_path(parse_image_capabilities(self.caps)).keys(),
        )

    def test_no_namespace(self):
        xml = b'<ImageChannel><Gain><GainLevel min="0" max="9">3</GainLevel></Gain></ImageChannel>'
        (gain,) = parse_image_capabilities(xml)
        self.assertEqual((gain.path, gain.type), ("Gain/GainLevel", ImageSettingType.NUMBER))

    def test_switch_detection_is_case_insensitive(self):
        xml = b'<C><X><enabled opt="True,False">true</enabled></X></C>'
        (switch,) = parse_image_capabilities(xml)
        self.assertEqual(switch.type, ImageSettingType.SWITCH)

    def test_non_numeric_range_is_ignored(self):
        xml = b'<C><Gain><GainLevel min="a" max="b">1</GainLevel></Gain></C>'
        self.assertEqual(parse_image_capabilities(xml), [])

    def test_invalid_xml_raises_parse_error(self):
        import xml.etree.ElementTree as ET

        with self.assertRaises(ET.ParseError):
            parse_image_capabilities(b"not xml")


class TestParseImageValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pairs = load_fixture_pairs()
        cls.values = pairs[0][2] if pairs else b""

    def test_flattens_leaves(self):
        values = parse_image_values(self.values)
        self.assertEqual(values["Gain/GainLevel"], "45")
        self.assertEqual(values["NoiseReduce/GeneralMode/generalLevel"], "40")

    def test_skips_empty_and_container_elements(self):
        values = parse_image_values(self.values)
        self.assertNotIn("LaserLight", values)
        self.assertNotIn("Gain", values)


class TestISAPIClientImage(unittest.TestCase):
    def setUp(self):
        self.client = ISAPIClient(host=DEFAULT_HOST, username=DEFAULT_USERNAME, password="pw")
        pairs = load_fixture_pairs()
        self.caps = pairs[0][1] if pairs else b""
        self.values = pairs[0][2] if pairs else b""

    def _responses(self, *responses):
        return patch.object(self.client, "request_raw", side_effect=list(responses))

    def test_get_image_settings(self):
        with self._responses(
            RawResponse(200, self.caps), RawResponse(200, self.values)
        ) as request_raw:
            settings = self.client.get_image_settings(channel=1)
        self.assertEqual(len(settings), 30)
        self.assertEqual(
            [c.args for c in request_raw.call_args_list],
            [
                ("GET", "/ISAPI/Image/channels/1/capabilities"),
                ("GET", "/ISAPI/Image/channels/1"),
            ],
        )

    def test_channel_is_used_in_endpoint(self):
        with self._responses(
            RawResponse(200, self.caps), RawResponse(200, self.values)
        ) as request_raw:
            self.client.get_image_settings(channel=2)
        self.assertEqual(request_raw.call_args_list[0].args[1], "/ISAPI/Image/channels/2/capabilities")

    def test_status_codes_map_to_errors(self):
        cases = {
            401: ISAPIAuthError,
            403: ISAPIAuthError,
            404: ISAPINotFoundError,
            500: ISAPIError,
        }
        for status, error in cases.items():
            with self.subTest(status=status):
                with self._responses(RawResponse(status, b"<ResponseStatus/>")):
                    with self.assertRaises(error):
                        self.client.get_image_settings()

    def test_invalid_xml_raises_isapi_error(self):
        with self._responses(RawResponse(200, b"junk"), RawResponse(200, b"junk")):
            with self.assertRaises(ISAPIError):
                self.client.get_image_settings()

    def test_get_image_values(self):
        with self._responses(RawResponse(200, self.values)) as request_raw:
            values = self.client.get_image_values()
        self.assertEqual(values["Shutter/ShutterLevel"], "1/1000")
        request_raw.assert_called_once_with("GET", "/ISAPI/Image/channels/1")

    def test_get_image_values_invalid_xml(self):
        with self._responses(RawResponse(200, b"junk")):
            with self.assertRaises(ISAPIError):
                self.client.get_image_values()

    def test_set_image_setting(self):
        """Test sending an update for an image setting."""
        with self._responses(RawResponse(200, b"<ResponseStatus><statusCode>1</statusCode></ResponseStatus>")) as request_raw:
            self.client.set_image_setting(channel=1, setting_path="Color/brightnessLevel", value="85")

        request_raw.assert_called_once()
        args, _ = request_raw.call_args
        self.assertIn("PUT", args[0].upper())
        self.assertIn("/ISAPI/Image/channels/1", args[1])

    def test_set_image_setting_error_handling(self):
        """Test that server rejection during an update correctly raises ISAPIError."""
        with self._responses(RawResponse(400, b"<ResponseStatus><statusCode>4</statusCode></ResponseStatus>")):
            with self.assertRaises(ISAPIError):
                self.client.set_image_setting(channel=1, setting_path="Color/brightnessLevel", value="9999")

    def test_get_image_settings_missing_values_fallback(self):
        """Test that if fetching the values document fails (e.g. 404), it falls back to capabilities defaults."""
        with self._responses(
            RawResponse(200, self.caps), RawResponse(404, b"<ResponseStatus/>")
        ):
            settings = self.client.get_image_settings(channel=1)
        self.assertEqual(len(settings), 30)

    def test_get_image_settings_empty_capabilities_xml(self):
        """Test that empty or malformed XML for capabilities raises an appropriate error."""
        with self._responses(RawResponse(200, b""), RawResponse(200, self.values)):
            with self.assertRaises(ISAPIError):
                self.client.get_image_settings(channel=1)


if __name__ == "__main__":
    unittest.main()

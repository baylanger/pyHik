"""
pyhik.dump
~~~~~~~~~~
Save a device's image setting documents exactly as it sends them.

Run ``python -m pyhik.dump`` and answer the prompts (hostname or IP
address, port, username and password; the password is not echoed). Port 80 is
plain HTTP and port 443 is HTTPS; for any other port it asks which one the
device uses, and for a certificate it cannot verify (cameras often have a
self-signed one) it asks whether to go on without checking. For a camera,
or for every camera behind an NVR, it saves the image capabilities and
current values documents, plus the channel list, to a new folder in the
current directory. ``--with-events`` also saves the documents the event
side of pyhik reads (triggers, their capabilities, the video channel list and
the motion detection settings), and ``--stream`` records the device's live
event stream for 60 seconds (or the number of seconds given). Only read
requests are sent; nothing on the device is changed.

The files are what pyhik parses, so they are useful to report a problem with a
model or to add it to the test fixtures. File names carry the model and the
firmware version (``DS-2CD2385G1-I-V5.6.3``), so two firmware versions of one
model do not overwrite each other. A ``device-<name>.txt`` summary lists the
model, device type and firmware; the serial number, MAC address, device name
and device ID are left out. The image documents are saved byte for byte and
hold settings only. In the event documents and the event stream, addresses,
user names, passwords and channel names are replaced with ``REDACTED``, and
pictures attached to events are left out. Look through the files before
sharing them.

Licensed under the MIT license.
"""

import argparse
from datetime import datetime
import getpass
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import warnings
import xml.etree.ElementTree as ET

import requests
from urllib3.exceptions import InsecureRequestWarning

from pyhik.constants import __version__
from pyhik.isapi import (
    ENDPOINT_DEVICE_INFO,
    REQUEST_TIMEOUT,
    ISAPIClient,
    ISAPIConnectionError,
    RawResponse,
    parse_sub_status,
)

# Image settings of a camera, and of the cameras attached to an NVR.
CAMERA_CHANNELS = "/ISAPI/Image/channels"
NVR_CHANNELS = "/ISAPI/ContentMgmt/ImageProxy/channels"

# Device info fields copied to device-<name>.txt. Everything else (serial number,
# MAC address, device name, device ID, ...) is left out.
SUMMARY_FIELDS = ("model", "deviceType", "firmwareVersion", "firmwareReleasedDate")

# Documents the event side of pyhik (HikCamera) reads, in the order it tries
# them: (file name stem, endpoints, optional). An optional document that a
# device does not offer is noted but not counted as a failure.
EVENT_DOCUMENTS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("event_triggers", ("/ISAPI/Event/triggers", "/Event/triggers"), False),
    ("event_triggers_cap", ("/ISAPI/Event/triggersCap",), True),
    (
        "video_channels",
        (
            "/ISAPI/System/Video/inputs/channels",
            "/ISAPI/ContentMgmt/InputProxy/channels",
            "/ISAPI/Streaming/channels",
        ),
        False,
    ),
    ("motion_detection", ("/ISAPI/System/Video/inputs/channels/1/motionDetection",), True),
)

STREAM_ENDPOINTS = ("/ISAPI/Event/notification/alertStream", "/Event/notification/alertStream")
DEFAULT_STREAM_SECONDS = 60
MAX_FIRMWARE_NAME = 40
DEFAULT_HTTP_PORT = 80
DEFAULT_HTTPS_PORT = 443

# Elements whose text is replaced in the event documents and the stream.
REDACTED_ELEMENTS = (
    "ipAddress", "ipv6Address", "ipV6Address", "macAddress", "hostName",
    "userName", "password", "serialNumber", "deviceID", "deviceName", "channelName",
    "name",  # channel names, which often say where a camera is
)
_REDACT = re.compile(
    r"<((?:[\w.-]+:)?(?:%s))(\s[^>]*)?>[^<]*</\1>" % "|".join(REDACTED_ELEMENTS)
)
_ALERT = re.compile(r"<EventNotificationAlert\b.*?</EventNotificationAlert>", re.S)

PASSWORD_ENV = "PYHIK_PASSWORD"

EXIT_OK = 0
EXIT_PARTIAL = 1  # connected, but some documents could not be read
EXIT_FAILED = 2  # could not connect, or the login was rejected


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local(child.tag) == name:
            return (child.text or "").strip()
    return ""


def parse_device_info(content: bytes) -> Dict[str, str]:
    """Return the summary fields of a deviceInfo document (empty if unreadable)."""
    try:
        root = ET.fromstring(content.strip())
    except ET.ParseError:
        return {}
    return {name: _child_text(root, name) for name in SUMMARY_FIELDS if _child_text(root, name)}


def parse_channel_ids(content: bytes) -> List[str]:
    """Return the IDs of the ``ImageChannel`` entries of a channel list, in order."""
    try:
        root = ET.fromstring(content.strip())
    except ET.ParseError:
        return []
    nodes = [root] if _local(root.tag) == "ImageChannel" else [
        e for e in root.iter() if _local(e.tag) == "ImageChannel"
    ]
    ids: List[str] = []
    for node in nodes:
        channel_id = _child_text(node, "id")
        if channel_id and channel_id not in ids:
            ids.append(channel_id)
    return ids


def is_nvr(device_type: str) -> bool:
    """Return True for a recorder (NVR, DVR, ...), whose cameras are reached by proxy."""
    upper = device_type.upper()
    return "NVR" in upper or "DVR" in upper


def detect_proxy(client: ISAPIClient, device_type: str, log: Callable[[str], None]) -> bool:
    """Return True if the cameras are reached through the NVR endpoints.

    The device type says NVR or DVR for most recorders, but some report the type of
    a camera (IPC, for example) or do not answer the device info request at all.
    Unless the type settles it, the camera endpoints are tried first, and if the
    device answers them with an error other than a refused login, the NVR endpoints.
    """
    if is_nvr(device_type):
        return True
    camera = client.request_raw("GET", CAMERA_CHANNELS)
    if camera.ok or camera.status_code in (401, 403):
        return False
    log("  %s: %s; trying the NVR endpoints" % (CAMERA_CHANNELS, _describe(camera)))
    return client.request_raw("GET", NVR_CHANNELS).ok


def _clean(text: str) -> str:
    """Replace every run of characters other than letters, digits, ``.``, ``_`` and ``-``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def safe_name(text: str) -> str:
    """Turn a model name into something safe for a file name."""
    return _clean(text) or "unknown"


def file_prefix(info: Dict[str, str]) -> str:
    """Return the model and firmware version, safe for file names.

    For example ``DS-2CD2385G1-I-V5.6.3``. A space, a ``/``, a quote or any other
    odd character becomes ``_`` (``v 1.0.4 beta 3`` is ``v_1.0.4_beta_3``), and the
    firmware part is cut at ``MAX_FIRMWARE_NAME`` characters. Without a firmware
    version it is the model alone.
    """
    firmware = _clean(info.get("firmwareVersion", ""))[:MAX_FIRMWARE_NAME].strip("_")
    model = safe_name(info.get("model", ""))
    return "%s-%s" % (model, firmware) if firmware else model


def redact(text: str) -> str:
    """Replace addresses, user names, passwords and names with REDACTED."""
    return _REDACT.sub(lambda m: "<%s%s>REDACTED</%s>" % (m.group(1), m.group(2) or "", m.group(1)), text)


def extract_alerts(content: bytes) -> Tuple[List[str], int]:
    """Return the event XML blocks of an alert stream, and how many pictures it held.

    The blocks are found the way pyhik reads the stream, so multipart and
    plain streams both work. Picture parts (``Content-Type: image/...``) are
    only counted.
    """
    text = content.decode("utf-8", "ignore")
    pictures = len(re.findall(r"content-type:\s*image/", text, re.I))
    return _ALERT.findall(text), pictures


def _open_stream(client: ISAPIClient, endpoint: str, read_timeout: float) -> Any:
    """Open the alert stream with the client's session and authentication."""
    client._detect_auth_method()  # pylint: disable=protected-access
    try:
        return client._session.get(  # pylint: disable=protected-access
            "%s%s" % (client.base_url, endpoint),
            auth=client._auth,  # pylint: disable=protected-access
            stream=True,
            timeout=(REQUEST_TIMEOUT, read_timeout),
        )
    except requests.exceptions.RequestException as err:
        raise ISAPIConnectionError("Cannot open the event stream: %s" % err) from err


def record_stream(
    open_stream: Callable[[str, float], Any],
    seconds: float,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[Optional[bytes], str]:
    """Record the alert stream for ``seconds``.

    Returns the bytes received (None if it could not be opened) and the
    endpoint used or the reason it failed. The recording also ends when the
    device closes the stream or stays silent past the end.
    """
    response = None
    endpoint = STREAM_ENDPOINTS[0]
    for endpoint in STREAM_ENDPOINTS:
        response = open_stream(endpoint, seconds + 5)
        if response.status_code != 404:
            break
        response.close()
    if response is None or response.status_code != 200:
        status = response.status_code if response is not None else 0
        return None, "%s: HTTP %d" % (endpoint, status)

    deadline = clock() + seconds
    chunks: List[bytes] = []
    try:
        # Small reads, like pyhik's own stream reader, so the end time is noticed
        # even when the device does not use chunked transfer encoding.
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                chunks.append(chunk)
            if clock() >= deadline:
                break
    except requests.exceptions.RequestException:
        pass  # silence past the end, or the device closed the stream: keep what arrived
    finally:
        response.close()
    return b"".join(chunks), endpoint


def _describe(response: RawResponse) -> str:
    sub_status = parse_sub_status(response.content)
    return "HTTP %d%s" % (response.status_code, " (%s)" % sub_status if sub_status else "")


def dump(
    client: ISAPIClient,
    output: Path,
    proxy: Optional[bool] = None,
    channels: Optional[Sequence[str]] = None,
    log: Callable[[str], None] = print,
    with_events: bool = False,
    stream_seconds: Optional[float] = None,
    open_stream: Optional[Callable[[str, float], Any]] = None,
) -> int:
    """Save the device's image documents to ``output``. Returns an exit code.

    Args:
        client: A connected client.
        output: Folder for the files; created if needed.
        proxy: True for an NVR, False for a camera, None to detect it: from the
            device type, and if that does not say NVR or DVR, by which of the
            two sets of endpoints answers.
        channels: Channel IDs to save; None for all listed channels.
        log: Where progress messages go.
        with_events: Also save the documents the event side of pyhik reads.
        stream_seconds: Also record the live event stream for this long.
        open_stream: Opens the event stream; the client's session by default.
    """
    info_response = client.request_raw("GET", ENDPOINT_DEVICE_INFO)
    if info_response.status_code == 401:
        log("The device rejected the username or password.")
        return EXIT_FAILED
    info = parse_device_info(info_response.content) if info_response.ok else {}
    if not info_response.ok:
        log("Could not read the device info: %s" % _describe(info_response))

    model = info.get("model", "")
    device_type = info.get("deviceType", "")
    prefix = file_prefix(info)
    log("Device: %s, type %s, firmware %s" % (
        model or "unknown model",
        device_type or "unknown",
        info.get("firmwareVersion", "unknown"),
    ))
    if proxy is None:
        proxy = detect_proxy(client, device_type, log)
    base = NVR_CHANNELS if proxy else CAMERA_CHANNELS
    log("Reading %s image settings from %s" % ("NVR (proxied camera)" if proxy else "camera", base))

    output.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    problems: List[Tuple[str, str]] = []
    not_offered: List[str] = []
    notes: List[str] = []

    def save(endpoint: str, filename: str) -> Optional[bytes]:
        response = client.request_raw("GET", endpoint)
        if not response.ok:
            problems.append((endpoint, _describe(response)))
            log("  %s: %s" % (endpoint, _describe(response)))
            return None
        (output / filename).write_bytes(response.content)
        saved.append(filename)
        log("  %s -> %s" % (endpoint, filename))
        return response.content

    def save_first(endpoints: Sequence[str], filename: str, optional: bool) -> None:
        """Save the first endpoint that answers, redacted."""
        for endpoint in endpoints:
            response = client.request_raw("GET", endpoint)
            if response.ok:
                (output / filename).write_text(redact(response.text), encoding="utf-8")
                saved.append(filename)
                notes.append("%s: %s" % (filename, endpoint))
                log("  %s -> %s" % (endpoint, filename))
                return
            log("  %s: %s" % (endpoint, _describe(response)))
        if optional:
            not_offered.append(endpoints[-1])
        else:
            problems.append((endpoints[-1], _describe(response)))

    listing = save(base, "image_channels_list-%s.xml" % prefix)
    if channels is None:
        channels = parse_channel_ids(listing) if listing is not None else []
        if not channels and not proxy:
            # Some cameras do not list their channels; the first one is 1.
            channels = ["1"]
    if not channels:
        log("No channels to read.")

    for channel in channels:
        endpoint = "%s/%s" % (base, channel)
        save("%s/capabilities" % endpoint, "image_capabilities-%s-ch%s.xml" % (prefix, safe_name(channel)))
        save(endpoint, "image_values-%s-ch%s.xml" % (prefix, safe_name(channel)))

    if with_events:
        log("Reading the event documents")
        for stem, endpoints, optional in EVENT_DOCUMENTS:
            save_first(endpoints, "%s-%s.xml" % (stem, prefix), optional)

    if stream_seconds:
        log("Recording the event stream for %g seconds (trigger some events now)" % stream_seconds)
        opener = open_stream or (lambda endpoint, timeout: _open_stream(client, endpoint, timeout))
        content, used = record_stream(opener, stream_seconds)
        if content is None:
            problems.append(("event stream", used))
            log("  event stream: %s" % used)
        else:
            alerts, pictures = extract_alerts(content)
            filename = "alert_stream-%s.txt" % prefix
            (output / filename).write_text(
                "".join(redact(alert) + "\n\n" for alert in alerts), encoding="utf-8"
            )
            saved.append(filename)
            notes.append(
                "%s: %s, %g s, %d event(s), %d picture(s) left out"
                % (filename, used, stream_seconds, len(alerts), pictures)
            )
            log("  %s -> %s (%d events)" % (used, filename, len(alerts)))

    lines = ["%s: %s" % (name, info[name]) for name in SUMMARY_FIELDS if name in info]
    lines += [
        "endpoints: %s" % ("NVR proxy" if proxy else "camera"),
        "channels: %s" % (", ".join(channels) or "none"),
        "pyhik: %s" % __version__,
        "captured: %s" % datetime.now().astimezone().isoformat(timespec="seconds"),
    ]
    lines += notes
    lines += ["not offered: %s" % endpoint for endpoint in not_offered]
    lines += ["not read: %s (%s)" % problem for problem in problems]
    summary = "device-%s.txt" % prefix
    (output / summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("Saved %d file(s) and %s in %s" % (len(saved), summary, output.resolve()))
    log(
        "Look through the files, then attach them to an issue at "
        "https://github.com/mezz64/pyHik/issues or add them to test/fixtures/ in a pull request."
    )
    return EXIT_PARTIAL if problems else EXIT_OK


def _ask(prompt: str) -> str:
    """Read one answer; an empty one when there is no more input."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def choose_connection(
    host: str,
    port: Optional[int],
    use_ssl: bool,
    ask: Callable[[str], str] = _ask,
) -> Tuple[str, int, bool]:
    """Return ``(host, port, use_ssl)``, asking only for what is not known.

    ``host`` may be a bare name or address, ``name:port`` or a URL such as
    ``https://nvr.local:8443``. A scheme or a port in it counts as given. With no
    port given the answer to "Port" is asked (80 if left empty). Port 80 means
    HTTP and 443 means HTTPS; for any other port the answer to "Use HTTPS" is
    asked. ``--ssl`` alone means port 443.

    Raises:
        ValueError: The host or its port cannot be read.
    """
    text = host.strip()
    parsed = urlparse(text if "://" in text else "//" + text)
    name = parsed.hostname or ""
    if parsed.port is not None:
        port = port or parsed.port
    if parsed.scheme == "https":
        use_ssl = True
    scheme_known = use_ssl or parsed.scheme == "http"

    if port is None:
        if use_ssl:
            port = DEFAULT_HTTPS_PORT
        elif scheme_known:
            port = DEFAULT_HTTP_PORT
        else:
            while port is None:
                answer = ask("Port [%d]: " % DEFAULT_HTTP_PORT)
                if not answer:
                    port = DEFAULT_HTTP_PORT
                elif answer.isdigit() and 0 < int(answer) < 65536:
                    port = int(answer)
                else:
                    print("The port is a number from 1 to 65535.", file=sys.stderr)

    if not scheme_known:
        if port == DEFAULT_HTTPS_PORT:
            use_ssl = True
        elif port != DEFAULT_HTTP_PORT:
            use_ssl = ask("Does the device use HTTPS on port %d? [y/N]: " % port).lower() in ("y", "yes")
    return name, port, use_ssl


def _causes(err: BaseException) -> List[BaseException]:
    """Return ``err`` and the exceptions it was raised from, innermost last."""
    chain: List[BaseException] = []
    while err is not None and err not in chain:
        chain.append(err)
        err = err.__cause__ or err.__context__
    return chain


def certificate_not_trusted(err: BaseException) -> bool:
    """True when a connection failed because the HTTPS certificate cannot be verified."""
    return any(
        isinstance(cause, requests.exceptions.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(cause)
        for cause in _causes(err)
    )


def tls_failed(err: BaseException) -> bool:
    """True when a connection failed in the TLS handshake (not a certificate check)."""
    return any(isinstance(cause, requests.exceptions.SSLError) for cause in _causes(err))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m pyhik.dump",
        description="Save a Hikvision device's image setting documents (read-only).",
    )
    parser.add_argument(
        "--host",
        help="hostname or IP address, optionally with a scheme and port (asked if missing)",
    )
    parser.add_argument("--username", help="user name (asked if missing)")
    parser.add_argument("--port", type=int, default=None, help="port (asked if missing; 443 with --ssl)")
    parser.add_argument("--ssl", action="store_true", help="use HTTPS (asked for ports other than 80 and 443)")
    parser.add_argument(
        "--no-verify", action="store_true",
        help="do not verify the HTTPS certificate (otherwise asked when it cannot be verified)",
    )
    parser.add_argument("--channel", action="append", help="only this channel ID (repeatable)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nvr", dest="proxy", action="store_const", const=True,
                       help="read the cameras behind an NVR (detected automatically)")
    group.add_argument("--camera", dest="proxy", action="store_const", const=False,
                       help="read a camera's own settings (detected automatically)")
    parser.add_argument(
        "--with-events", action="store_true",
        help="also save the event documents (triggers, video channels, motion detection)",
    )
    parser.add_argument(
        "--stream", nargs="?", type=int, const=DEFAULT_STREAM_SECONDS, default=None,
        metavar="SECONDS",
        help="also record the live event stream (default %d seconds)" % DEFAULT_STREAM_SECONDS,
    )
    parser.add_argument(
        "--output", type=Path,
        help="folder for the files (default: a new pyhik-dump-<host>-<time> folder in the current directory)",
    )
    args = parser.parse_args(argv)
    if args.stream is not None and args.stream <= 0:
        parser.error("--stream needs a number of seconds above 0")
    if args.port is not None and not 0 < args.port < 65536:
        parser.error("--port needs a number from 1 to 65535")

    host = args.host or _ask("Hostname or IP address: ")
    if not host:
        print("A hostname or IP address is needed.", file=sys.stderr)
        return EXIT_FAILED
    try:
        host, port, use_ssl = choose_connection(host, args.port, args.ssl)
    except ValueError:
        print("Cannot read the hostname or port.", file=sys.stderr)
        return EXIT_FAILED
    username = args.username or _ask("Username: ")
    password = os.environ.get(PASSWORD_ENV) or getpass.getpass("Password: ")
    if not username:
        print("A username is needed.", file=sys.stderr)
        return EXIT_FAILED

    output = args.output or Path(
        "pyhik-dump-%s-%s" % (safe_name(host), datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    verify = not args.no_verify
    while True:
        if use_ssl and not verify:
            # requests lets these override a session's verify=False.
            for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
                os.environ.pop(name, None)
        client = ISAPIClient(host, port, username, password, use_ssl, verify)
        try:
            with warnings.catch_warnings():
                if use_ssl and not verify:
                    warnings.simplefilter("ignore", InsecureRequestWarning)
                return dump(
                    client,
                    output,
                    proxy=args.proxy,
                    channels=args.channel,
                    with_events=args.with_events,
                    stream_seconds=args.stream,
                )
        except ISAPIConnectionError as err:
            if use_ssl and verify and certificate_not_trusted(err):
                print("The certificate of %s could not be verified (a self-signed certificate is common on cameras)." % host)
                if _ask("Go on without checking it? [y/N]: ").lower() in ("y", "yes"):
                    verify = False
                    continue
                print("Not going on. Run again and answer yes, or add --no-verify, to skip the check.", file=sys.stderr)
                return EXIT_FAILED
            print("Could not reach %s:%d: %s" % (host, port, err), file=sys.stderr)
            if use_ssl and tls_failed(err):
                print("The device did not answer HTTPS on port %d; check the port, or answer no to HTTPS." % port, file=sys.stderr)
            elif not use_ssl:
                print("If the device only accepts HTTPS, run again with port %d." % DEFAULT_HTTPS_PORT, file=sys.stderr)
            return EXIT_FAILED
        finally:
            client.close()


if __name__ == "__main__":
    sys.exit(main())

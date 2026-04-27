#!/usr/bin/env python3
"""Forward NTRIP RTCM corrections to a serial RTK receiver.

The script connects to an NTRIP caster, writes RTCM bytes into a serial
receiver port, and optionally sends the latest GGA sentence back upstream for
VRS services.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import os
import selectors
import signal
import socket
import sys
import termios
import time


BAUD_MAP = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: termios.B230400,
    460800: termios.B460800,
    921600: termios.B921600,
}


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to an NTRIP caster and feed RTCM to an RTK receiver."
    )
    parser.add_argument(
        "--config",
        help="Optional ini config file with [SYSTEM] and [RTK] sections",
    )
    parser.add_argument("--host", help="NTRIP caster hostname or IP")
    parser.add_argument("--port", type=int, help="NTRIP caster port")
    parser.add_argument("--mountpoint", help="NTRIP mountpoint name")
    parser.add_argument("--username", help="NTRIP username")
    parser.add_argument("--password", help="NTRIP password")
    parser.add_argument("--serial-device", help="Serial port connected to the RTK rover")
    parser.add_argument(
        "--baud",
        type=int,
        choices=sorted(BAUD_MAP),
        help="Serial baud rate",
    )
    parser.add_argument(
        "--gga-interval",
        type=float,
        default=10.0,
        help="Seconds between upstream GGA sentences",
    )
    parser.add_argument(
        "--no-gga",
        action="store_true",
        help="Do not send GGA upstream to the caster",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=15.0,
        help="Socket timeout in seconds",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        help="Seconds between status logs",
    )
    parser.add_argument(
        "--um980-status-interval",
        type=float,
        default=0.0,
        help="Seconds between UM980 RTKSTATUSA queries; 0 disables the query",
    )
    return parser


def read_ini_config(path: str) -> dict[str, object]:
    parser = configparser.ConfigParser()
    read_files = parser.read(path, encoding="utf-8")
    if not read_files:
        raise RuntimeError(f"Config file not found: {path}")

    values: dict[str, object] = {}
    if parser.has_option("SYSTEM", "gps_serial"):
        values["serial_device"] = parser.get("SYSTEM", "gps_serial").strip()
    if parser.has_option("SYSTEM", "gps_rate"):
        values["baud"] = parser.getint("SYSTEM", "gps_rate")
    if parser.has_option("RTK", "ntrip_ip"):
        values["host"] = parser.get("RTK", "ntrip_ip").strip()
    if parser.has_option("RTK", "ntrip_port"):
        values["port"] = parser.getint("RTK", "ntrip_port")
    if parser.has_option("RTK", "ntrip_point"):
        values["mountpoint"] = parser.get("RTK", "ntrip_point").strip()
    if parser.has_option("RTK", "ntrip_user"):
        values["username"] = parser.get("RTK", "ntrip_user").strip()
    if parser.has_option("RTK", "ntrip_pwd"):
        values["password"] = parser.get("RTK", "ntrip_pwd").strip()
    return values


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_values: dict[str, object] = {}
    if args.config:
        args.config = os.path.abspath(args.config)
        config_values = read_ini_config(args.config)

    defaults = {
        "port": 2101,
        "serial_device": "/dev/ttyrtk",
        "baud": 115200,
    }
    fields = [
        "host",
        "port",
        "mountpoint",
        "username",
        "password",
        "serial_device",
        "baud",
    ]

    for field in fields:
        if getattr(args, field) is None and field in config_values:
            setattr(args, field, config_values[field])
        if getattr(args, field) is None and field in defaults:
            setattr(args, field, defaults[field])

    if args.baud not in BAUD_MAP:
        parser.error(
            f"unsupported baud rate {args.baud}; choose one of {sorted(BAUD_MAP)}"
        )

    missing = [
        name
        for name in ("host", "mountpoint", "username", "password")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(
            "missing required values: "
            + ", ".join(missing)
            + " (set them via CLI or --config)"
        )

    return args


def configure_serial(fd: int, baud: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = BAUD_MAP[baud]
    attrs[5] = BAUD_MAP[baud]
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


def open_serial(path: str, baud: int) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    configure_serial(fd, baud)
    return fd


def connect_caster(args: argparse.Namespace) -> tuple[socket.socket, bytes]:
    mountpoint = args.mountpoint.lstrip("/")
    credentials = f"{args.username}:{args.password}".encode("utf-8")
    auth = base64.b64encode(credentials).decode("ascii")

    request = "\r\n".join(
        [
            f"GET /{mountpoint} HTTP/1.0",
            "User-Agent: NTRIP codex-rtk-bridge/1.0",
            f"Authorization: Basic {auth}",
            "Accept: */*",
            "",
            "",
        ]
    ).encode("ascii")

    sock = socket.create_connection((args.host, args.port), timeout=args.socket_timeout)
    sock.settimeout(args.socket_timeout)
    sock.sendall(request)

    first_chunk = sock.recv(4096)
    if not first_chunk:
        raise RuntimeError("Caster closed the connection before sending a response")

    if b"\r\n\r\n" in first_chunk:
        raw_header, initial_payload = first_chunk.split(b"\r\n\r\n", 1)
        status_line = raw_header.splitlines()[0].decode("ascii", errors="replace")
    else:
        first_line, _sep, rest = first_chunk.partition(b"\r\n")
        status_line = first_line.decode("ascii", errors="replace")
        initial_payload = rest

    if not (
        status_line.startswith("ICY 200")
        or status_line.startswith("HTTP/1.0 200")
        or status_line.startswith("HTTP/1.1 200")
    ):
        raise RuntimeError(f"NTRIP caster rejected the request: {status_line}")

    log(f"Connected to caster: {status_line}")
    return sock, initial_payload


def extract_gga(line: bytes) -> str | None:
    line = line.strip()
    if not line.startswith(b"$") or b"GGA" not in line:
        return None
    text = line.decode("ascii", errors="ignore")
    parts = text.split(",")
    if len(parts) < 10:
        return None
    return text


def gga_summary(gga: str) -> str:
    parts = gga.split(",")
    if len(parts) < 9:
        return "GGA available"
    quality = parts[6] or "?"
    sats = parts[7] or "?"
    hdop = parts[8] or "?"
    age = parts[13] if len(parts) > 13 else ""
    station = parts[14].split("*", 1)[0] if len(parts) > 14 else ""
    summary = f"fix={quality} sats={sats} hdop={hdop}"
    if age:
        summary += f" age={age}"
    if station:
        summary += f" station={station}"
    return summary


def extract_um980_rtkstatus(line: bytes) -> str | None:
    text = line.strip().decode("ascii", errors="ignore")
    if not text.startswith("#RTKSTATUSA"):
        return None
    return text


def send_serial(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(argv)
    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    serial_fd = -1
    sock: socket.socket | None = None
    selector = selectors.DefaultSelector()

    try:
        if args.config:
            log(f"Loaded config from {args.config}")
        log(
            f"Using {args.serial_device} @ {args.baud}, mountpoint={args.mountpoint}, "
            f"caster={args.host}:{args.port}"
        )

        serial_fd = open_serial(args.serial_device, args.baud)
        log(f"Opened serial port {args.serial_device} at {args.baud} baud")

        sock, initial_payload = connect_caster(args)
        if initial_payload:
            send_serial(serial_fd, initial_payload)

        selector.register(sock, selectors.EVENT_READ, data="caster")
        selector.register(serial_fd, selectors.EVENT_READ, data="serial")

        rx_buffer = bytearray()
        last_gga: str | None = None
        next_gga_at = time.monotonic()
        last_status_at = time.monotonic()
        next_um980_status_at = time.monotonic()
        total_rtcm = len(initial_payload)
        total_gga = 0
        total_um980_queries = 0
        last_um980_status: str | None = None

        while not stop:
            timeout = 0.2
            if not args.no_gga and last_gga:
                timeout = max(0.0, min(timeout, next_gga_at - time.monotonic()))
            if args.um980_status_interval > 0:
                timeout = max(
                    0.0, min(timeout, next_um980_status_at - time.monotonic())
                )

            for key, _mask in selector.select(timeout):
                if key.data == "caster":
                    data = sock.recv(4096)
                    if not data:
                        raise RuntimeError("NTRIP caster closed the stream")
                    send_serial(serial_fd, data)
                    total_rtcm += len(data)
                else:
                    incoming = os.read(serial_fd, 4096)
                    if incoming:
                        rx_buffer.extend(incoming.replace(b"\r", b"\n"))
                        while b"\n" in rx_buffer:
                            raw_line, _, rx_buffer = rx_buffer.partition(b"\n")
                            gga = extract_gga(raw_line)
                            if gga:
                                last_gga = gga
                            um980_status = extract_um980_rtkstatus(raw_line)
                            if um980_status:
                                last_um980_status = um980_status
                                log(f"UM980 {um980_status}")

            now = time.monotonic()
            if not args.no_gga and last_gga and now >= next_gga_at:
                sock.sendall(last_gga.encode("ascii") + b"\r\n")
                total_gga += 1
                next_gga_at = now + args.gga_interval
                log(f"Sent GGA upstream ({gga_summary(last_gga)})")

            if args.um980_status_interval > 0 and now >= next_um980_status_at:
                send_serial(serial_fd, b"RTKSTATUSA\r\n")
                total_um980_queries += 1
                next_um980_status_at = now + args.um980_status_interval

            if now - last_status_at >= args.status_interval:
                status = gga_summary(last_gga) if last_gga else "no GGA yet"
                message = (
                    f"Forwarded RTCM={total_rtcm} bytes, sent GGA={total_gga}, "
                    f"sent UM980 queries={total_um980_queries}, latest {status}"
                )
                if last_um980_status:
                    message += f", last RTKSTATUSA seen"
                log(message)
                last_status_at = now

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log(f"Bridge stopped: {exc}")
        return 1
    finally:
        try:
            selector.close()
        except Exception:
            pass
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if serial_fd >= 0:
            try:
                os.close(serial_fd)
            except Exception:
                pass
        log("Exited cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())

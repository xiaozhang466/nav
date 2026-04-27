#!/usr/bin/env python3
"""Monitor UM980 serial output without connecting to NTRIP."""

from __future__ import annotations

import argparse
import configparser
import os
import selectors
import signal
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
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor UM980 GGA/RTKSTATUSA from a serial port."
    )
    parser.add_argument(
        "--config",
        help="Optional ini config file with [SYSTEM] section",
    )
    parser.add_argument("--serial-device", help="Serial port connected to the UM980")
    parser.add_argument(
        "--baud",
        type=int,
        choices=sorted(BAUD_MAP),
        help="Serial baud rate",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        help="Seconds between summary logs",
    )
    parser.add_argument(
        "--um980-status-interval",
        type=float,
        default=5.0,
        help="Seconds between RTKSTATUSA queries; 0 disables the query",
    )
    return parser


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_values: dict[str, object] = {}
    if args.config:
        args.config = os.path.abspath(args.config)
        config_values = read_ini_config(args.config)

    defaults = {
        "serial_device": "/dev/ttyrtk",
        "baud": 115200,
    }
    fields = ["serial_device", "baud"]
    for field in fields:
        if getattr(args, field) is None and field in config_values:
            setattr(args, field, config_values[field])
        if getattr(args, field) is None and field in defaults:
            setattr(args, field, defaults[field])

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


def send_serial(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


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


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(argv)
    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    fd = -1
    selector = selectors.DefaultSelector()

    try:
        if args.config:
            log(f"Loaded config from {args.config}")
        log(f"Monitoring {args.serial_device} @ {args.baud}")

        fd = open_serial(args.serial_device, args.baud)
        selector.register(fd, selectors.EVENT_READ)

        rx_buffer = bytearray()
        last_gga: str | None = None
        last_gga_summary: str | None = None
        last_um980_status: str | None = None
        next_um980_status_at = time.monotonic()
        last_status_at = time.monotonic()

        while not stop:
            timeout = 0.2
            if args.um980_status_interval > 0:
                timeout = max(0.0, min(timeout, next_um980_status_at - time.monotonic()))

            events = selector.select(timeout)
            if events:
                incoming = os.read(fd, 4096)
                if incoming:
                    rx_buffer.extend(incoming.replace(b"\r", b"\n"))
                    while b"\n" in rx_buffer:
                        raw_line, _, rx_buffer = rx_buffer.partition(b"\n")
                        gga = extract_gga(raw_line)
                        if gga:
                            last_gga = gga
                            summary = gga_summary(gga)
                            if summary != last_gga_summary:
                                log(f"GGA {summary}")
                                last_gga_summary = summary
                        um980_status = extract_um980_rtkstatus(raw_line)
                        if um980_status:
                            last_um980_status = um980_status
                            log(f"UM980 {um980_status}")

            now = time.monotonic()
            if args.um980_status_interval > 0 and now >= next_um980_status_at:
                send_serial(fd, b"RTKSTATUSA\r\n")
                next_um980_status_at = now + args.um980_status_interval

            if now - last_status_at >= args.status_interval:
                summary = gga_summary(last_gga) if last_gga else "no GGA yet"
                message = f"Latest {summary}"
                if last_um980_status:
                    message += ", last RTKSTATUSA seen"
                log(message)
                last_status_at = now

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log(f"Monitor stopped: {exc}")
        return 1
    finally:
        try:
            selector.close()
        except Exception:
            pass
        if fd >= 0:
            try:
                os.close(fd)
            except Exception:
                pass
        log("Exited cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())

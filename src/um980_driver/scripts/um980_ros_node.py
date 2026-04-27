#!/usr/bin/env python3
"""ROS node that publishes UM980 RTK data as standard ROS topics.

The node listens to a UM980 serial stream, extracts GGA sentences, and
publishes them as ``sensor_msgs/NavSatFix``. It also exposes lightweight
status/debug topics so the RTK state can be inspected before sensor fusion
is added.
"""

from __future__ import annotations

import configparser
import math
import os
import re
import signal
import time
from dataclasses import dataclass

import rospy
import serial
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String, UInt8


SUPPORTED_BAUDS = (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600)


QUALITY_LABELS = {
    0: "no_fix",
    1: "gps_fix",
    2: "dgps",
    3: "pps",
    4: "rtk_fixed",
    5: "rtk_float",
    6: "dead_reckoning",
    7: "manual_input",
    8: "simulation",
    9: "waas",
}


TALKER_SERVICES = {
    "GP": NavSatStatus.SERVICE_GPS,
    "GL": NavSatStatus.SERVICE_GLONASS,
    "GA": NavSatStatus.SERVICE_GALILEO,
    "GB": NavSatStatus.SERVICE_COMPASS,
    "BD": NavSatStatus.SERVICE_COMPASS,
    "GN": (
        NavSatStatus.SERVICE_GPS
        | NavSatStatus.SERVICE_GLONASS
        | NavSatStatus.SERVICE_GALILEO
        | NavSatStatus.SERVICE_COMPASS
    ),
}


@dataclass
class GgaData:
    raw: str
    talker: str
    latitude: float
    longitude: float
    altitude: float
    quality: int
    satellites: int
    hdop: float | None
    age: float | None
    station_id: str


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
    if parser.has_option("GPS", "offset_yaw"):
        values["offset_yaw"] = parser.getfloat("GPS", "offset_yaw")
    return values


def open_serial(path: str, baud: int, timeout: float) -> serial.Serial:
    return serial.Serial(
        port=path,
        baudrate=baud,
        timeout=timeout,
        write_timeout=1.0,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )


def send_serial(port: serial.Serial, payload: bytes) -> None:
    port.write(payload)
    port.flush()


def verify_nmea_checksum(sentence: str) -> bool:
    if "*" not in sentence:
        return True
    body, checksum_text = sentence[1:].split("*", 1)
    checksum_text = checksum_text[:2]
    try:
        expected = int(checksum_text, 16)
    except ValueError:
        return False

    actual = 0
    for char in body:
        actual ^= ord(char)
    return actual == expected


def parse_nmea_coordinate(raw_value: str, hemisphere: str, is_latitude: bool) -> float:
    if not raw_value or not hemisphere:
        raise ValueError("missing coordinate field")

    degree_digits = 2 if is_latitude else 3
    degrees = float(raw_value[:degree_digits])
    minutes = float(raw_value[degree_digits:])
    value = degrees + minutes / 60.0

    if hemisphere in ("S", "W"):
        value = -value
    elif hemisphere not in ("N", "E"):
        raise ValueError(f"invalid hemisphere {hemisphere}")
    return value


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_gga_sentence(sentence: str) -> GgaData | None:
    sentence = sentence.strip()
    if not sentence.startswith("$") or "GGA" not in sentence:
        return None
    if not verify_nmea_checksum(sentence):
        raise ValueError(f"invalid NMEA checksum: {sentence}")

    body = sentence[1:].split("*", 1)[0]
    fields = body.split(",")
    if len(fields) < 15:
        raise ValueError(f"incomplete GGA sentence: {sentence}")

    message_type = fields[0]
    if not message_type.endswith("GGA"):
        return None

    talker = message_type[:2]
    quality = parse_int(fields[6], default=0)
    satellites = parse_int(fields[7], default=0)
    hdop = parse_float(fields[8])
    altitude_msl = parse_float(fields[9])
    geoid_separation = parse_float(fields[11])

    latitude = math.nan
    longitude = math.nan
    altitude = math.nan

    if quality > 0:
        latitude = parse_nmea_coordinate(fields[2], fields[3], is_latitude=True)
        longitude = parse_nmea_coordinate(fields[4], fields[5], is_latitude=False)
        if altitude_msl is not None and geoid_separation is not None:
            altitude = altitude_msl + geoid_separation
        elif altitude_msl is not None:
            altitude = altitude_msl

    age = parse_float(fields[13])
    station_id = fields[14]

    return GgaData(
        raw=sentence,
        talker=talker,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        quality=quality,
        satellites=satellites,
        hdop=hdop,
        age=age,
        station_id=station_id,
    )


def quality_to_label(quality: int) -> str:
    return QUALITY_LABELS.get(quality, f"unknown_{quality}")


def quality_to_navsat_status(quality: int) -> int:
    if quality <= 0:
        return NavSatStatus.STATUS_NO_FIX
    if quality in (2, 3, 4, 5, 9):
        return NavSatStatus.STATUS_GBAS_FIX
    return NavSatStatus.STATUS_FIX


def talker_to_service_mask(talker: str) -> int:
    return TALKER_SERVICES.get(talker, NavSatStatus.SERVICE_GPS)


def covariance_from_quality(quality: int, hdop: float | None) -> tuple[list[float], int]:
    if quality <= 0:
        return [0.0] * 9, NavSatFix.COVARIANCE_TYPE_UNKNOWN

    hdop_value = hdop if hdop is not None and hdop > 0.0 else 1.0
    base_sigma_xy = {
        1: 2.5,
        2: 0.8,
        3: 0.8,
        4: 0.05,
        5: 0.5,
        6: 5.0,
        7: 5.0,
        8: 5.0,
        9: 0.8,
    }.get(quality, 5.0)
    hdop_scale = {
        1: 1.5,
        2: 0.8,
        3: 0.8,
        4: 0.05,
        5: 0.4,
        6: 2.0,
        7: 2.0,
        8: 2.0,
        9: 0.8,
    }.get(quality, 2.0)

    sigma_xy = max(base_sigma_xy, hdop_value * hdop_scale)
    sigma_z = max(sigma_xy * 2.0, base_sigma_xy * 2.0)

    covariance = [0.0] * 9
    covariance[0] = sigma_xy * sigma_xy
    covariance[4] = sigma_xy * sigma_xy
    covariance[8] = sigma_z * sigma_z
    return covariance, NavSatFix.COVARIANCE_TYPE_APPROXIMATED


def parse_um980_rtkstatus(sentence: str) -> tuple[str, str] | None:
    sentence = sentence.strip()
    if not sentence.startswith("#RTKSTATUSA"):
        return None

    payload = sentence.split(";", 1)
    if len(payload) != 2:
        return sentence, ""

    fields = payload[1].split("*", 1)[0].split(",")
    raw_position_type = fields[11].strip() if len(fields) > 11 else ""

    # Occasionally a malformed serial line can splice the next NMEA sentence
    # into the RTKSTATUSA payload. Keep only clean UM980 position-type tokens.
    position_type = raw_position_type.split("$", 1)[0].split("#", 1)[0].strip()
    if position_type and not re.fullmatch(r"[A-Z0-9_]+", position_type):
        position_type = ""
    if position_type.endswith("_"):
        position_type = ""
    return sentence, position_type


class UM980RosNode:
    def __init__(self) -> None:
        config_path = rospy.get_param("~config", "")
        config_values: dict[str, object] = {}
        if config_path:
            config_path = os.path.abspath(config_path)
            config_values = read_ini_config(config_path)
            rospy.loginfo("Loaded UM980 config from %s", config_path)

        baud_default = int(config_values.get("baud", 115200))
        serial_default = str(config_values.get("serial_device", "/dev/ttyrtk"))

        self.serial_device = rospy.get_param("~serial_device", serial_default)
        self.baud = int(rospy.get_param("~baud", baud_default))
        self.frame_id = rospy.get_param("~frame_id", "gps_link")
        self.reconnect_delay = float(rospy.get_param("~reconnect_delay", 1.0))
        self.read_timeout = float(rospy.get_param("~read_timeout", 0.2))
        self.status_interval = float(rospy.get_param("~status_interval", 5.0))
        self.um980_status_interval = float(
            rospy.get_param("~um980_status_interval", 0.0)
        )
        self.bad_gga_warn_interval = float(
            rospy.get_param("~bad_gga_warn_interval", 30.0)
        )

        self.fix_topic = rospy.get_param("~fix_topic", "/rtk/fix")
        self.quality_topic = rospy.get_param("~quality_topic", "/rtk/fix_quality")
        self.fix_type_topic = rospy.get_param("~fix_type_topic", "/rtk/fix_type")
        self.gga_topic = rospy.get_param("~gga_topic", "/rtk/gga_raw")
        self.rtkstatus_topic = rospy.get_param(
            "~rtkstatus_topic", "/rtk/rtkstatus_raw"
        )
        self.position_type_topic = rospy.get_param(
            "~position_type_topic", "/rtk/position_type"
        )

        if self.baud not in SUPPORTED_BAUDS:
            raise ValueError(
                f"unsupported baud rate {self.baud}; choose one of {sorted(SUPPORTED_BAUDS)}"
            )

        self.fix_pub = rospy.Publisher(self.fix_topic, NavSatFix, queue_size=10)
        self.quality_pub = rospy.Publisher(self.quality_topic, UInt8, queue_size=10)
        self.fix_type_pub = rospy.Publisher(self.fix_type_topic, String, queue_size=10)
        self.gga_pub = rospy.Publisher(self.gga_topic, String, queue_size=20)
        self.rtkstatus_pub = rospy.Publisher(
            self.rtkstatus_topic, String, queue_size=20
        )
        self.position_type_pub = rospy.Publisher(
            self.position_type_topic, String, queue_size=20
        )

        self.last_gga: GgaData | None = None
        self.last_position_type = ""
        self.last_fix_label = ""
        self.last_summary_at = time.monotonic()

        rospy.loginfo(
            "UM980 ROS node configured: serial=%s baud=%d frame_id=%s",
            self.serial_device,
            self.baud,
            self.frame_id,
        )

    def publish_fix(self, gga: GgaData) -> None:
        now = rospy.Time.now()
        msg = NavSatFix()
        msg.header.stamp = now
        msg.header.frame_id = self.frame_id
        msg.status.status = quality_to_navsat_status(gga.quality)
        msg.status.service = talker_to_service_mask(gga.talker)

        if gga.quality > 0:
            msg.latitude = gga.latitude
            msg.longitude = gga.longitude
            msg.altitude = gga.altitude
            covariance, covariance_type = covariance_from_quality(
                gga.quality, gga.hdop
            )
            msg.position_covariance = covariance
            msg.position_covariance_type = covariance_type
        else:
            msg.latitude = math.nan
            msg.longitude = math.nan
            msg.altitude = math.nan
            msg.position_covariance = [0.0] * 9
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.fix_pub.publish(msg)
        self.quality_pub.publish(UInt8(data=max(0, gga.quality)))
        self.fix_type_pub.publish(String(data=quality_to_label(gga.quality)))
        self.gga_pub.publish(String(data=gga.raw))

        current_label = quality_to_label(gga.quality)
        if current_label != self.last_fix_label:
            rospy.loginfo(
                "RTK fix changed to %s (sats=%d hdop=%s)",
                current_label,
                gga.satellites,
                "n/a" if gga.hdop is None else f"{gga.hdop:.2f}",
            )
            self.last_fix_label = current_label

        self.last_gga = gga

    def publish_rtkstatus(self, raw_status: str, position_type: str) -> None:
        self.rtkstatus_pub.publish(String(data=raw_status))
        if position_type:
            self.position_type_pub.publish(String(data=position_type))
            if position_type != self.last_position_type:
                rospy.loginfo("UM980 position type changed to %s", position_type)
                self.last_position_type = position_type

    def log_periodic_summary(self) -> None:
        if self.status_interval <= 0.0:
            return
        now = time.monotonic()
        if now - self.last_summary_at < self.status_interval:
            return

        if self.last_gga is None:
            rospy.loginfo("Waiting for GGA data from UM980")
            self.last_summary_at = now
            return

        gga = self.last_gga
        lat = "nan" if math.isnan(gga.latitude) else f"{gga.latitude:.8f}"
        lon = "nan" if math.isnan(gga.longitude) else f"{gga.longitude:.8f}"
        altitude = "nan" if math.isnan(gga.altitude) else f"{gga.altitude:.3f}"
        age = "n/a" if gga.age is None else f"{gga.age:.1f}"
        hdop = "n/a" if gga.hdop is None else f"{gga.hdop:.2f}"
        station = gga.station_id or "n/a"
        rospy.loginfo(
            "Latest fix=%s lat=%s lon=%s alt=%s sats=%d hdop=%s age=%s station=%s",
            quality_to_label(gga.quality),
            lat,
            lon,
            altitude,
            gga.satellites,
            hdop,
            age,
            station,
        )
        self.last_summary_at = now

    def run(self) -> None:
        while not rospy.is_shutdown():
            serial_port: serial.Serial | None = None
            try:
                serial_port = open_serial(
                    self.serial_device, self.baud, self.read_timeout
                )
                rospy.loginfo(
                    "Opened UM980 serial port %s at %d baud",
                    self.serial_device,
                    self.baud,
                )

                rx_buffer = bytearray()
                next_um980_query_at = time.monotonic()

                while not rospy.is_shutdown():
                    incoming = serial_port.read(serial_port.in_waiting or 1)
                    if incoming:
                        rx_buffer.extend(incoming.replace(b"\r", b"\n"))
                        while b"\n" in rx_buffer:
                            raw_line, _, rx_buffer = rx_buffer.partition(b"\n")
                            line = raw_line.decode("ascii", errors="ignore").strip()
                            if not line:
                                continue

                            try:
                                gga = parse_gga_sentence(line)
                            except ValueError as exc:
                                if self.bad_gga_warn_interval > 0.0:
                                    rospy.logwarn_throttle(
                                        self.bad_gga_warn_interval,
                                        "Discarding bad GGA: %s",
                                        exc,
                                    )
                                gga = None

                            if gga is not None:
                                self.publish_fix(gga)

                            rtkstatus = parse_um980_rtkstatus(line)
                            if rtkstatus is not None:
                                raw_status, position_type = rtkstatus
                                self.publish_rtkstatus(raw_status, position_type)

                    now = time.monotonic()
                    if (
                        self.um980_status_interval > 0.0
                        and now >= next_um980_query_at
                    ):
                        send_serial(serial_port, b"RTKSTATUSA\r\n")
                        next_um980_query_at = now + self.um980_status_interval

                    self.log_periodic_summary()

            except (serial.SerialException, OSError) as exc:
                rospy.logwarn(
                    "UM980 serial connection failed on %s: %s",
                    self.serial_device,
                    exc,
                )
            except Exception as exc:
                rospy.logerr("UM980 node stopped by unexpected error: %s", exc)
            finally:
                if serial_port is not None:
                    try:
                        serial_port.close()
                    except Exception:
                        pass

            if not rospy.is_shutdown():
                rospy.sleep(self.reconnect_delay)


def main() -> int:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    rospy.init_node("um980_ros_node", anonymous=False)
    node = UM980RosNode()
    node.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

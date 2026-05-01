#!/usr/bin/env python3
"""High-rate TF diagnostic CSV logger.

This node records the actual transform values used by navigation so short
map->odom corrections can be correlated with controller oscillation.
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Dict, Optional, Tuple

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


TransformSample = Tuple[float, float, float]


def quaternion_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def format_float(value: Optional[float], fmt: str = "%.6f") -> str:
    if value is None:
        return ""
    return fmt % value


class TfDiagnosticLogger:
    def __init__(self) -> None:
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.lookup_timeout = float(rospy.get_param("~lookup_timeout", 0.01))
        self.tf_cache_time = float(rospy.get_param("~tf_cache_time", 10.0))
        self.flush_interval = float(rospy.get_param("~flush_interval", 1.0))
        self.log_dir = os.path.expanduser(
            rospy.get_param("~log_dir", os.path.join("~", ".ros", "nav_health"))
        )
        self.filename_prefix = rospy.get_param("~filename_prefix", "tf_diagnostics")

        self.transforms = [
            ("map_odom", self.map_frame, self.odom_frame),
            ("odom_base", self.odom_frame, self.base_frame),
            ("map_base", self.map_frame, self.base_frame),
        ]
        self.previous: Dict[str, TransformSample] = {}
        self.seq = 0
        self.rows_since_flush = 0
        self.flush_every_rows = max(1, int(max(self.rate_hz, 0.1) * max(self.flush_interval, 0.1)))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(max(self.tf_cache_time, 1.0)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        os.makedirs(self.log_dir, exist_ok=True)
        ros_time = rospy.Time.now().to_sec()
        suffix = str(int(ros_time if ros_time > 0.0 else time.time()))
        self.log_path = os.path.join(self.log_dir, "%s_%s.csv" % (self.filename_prefix, suffix))
        self.log_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.log_file, fieldnames=self.fieldnames())
        self.writer.writeheader()

        rospy.on_shutdown(self.close)
        rospy.loginfo(
            "tf diagnostic logger started: path=%s rate=%.1fHz frames=%s->%s,%s->%s,%s->%s",
            self.log_path,
            self.rate_hz,
            self.map_frame,
            self.odom_frame,
            self.odom_frame,
            self.base_frame,
            self.map_frame,
            self.base_frame,
        )

    def fieldnames(self):
        fields = ["time", "wall_time", "seq"]
        for prefix, _target, _source in self.transforms:
            fields.extend(
                [
                    "%s_ok" % prefix,
                    "%s_stamp" % prefix,
                    "%s_age" % prefix,
                    "%s_x" % prefix,
                    "%s_y" % prefix,
                    "%s_yaw" % prefix,
                    "%s_yaw_deg" % prefix,
                    "%s_xy_step" % prefix,
                    "%s_dyaw_deg" % prefix,
                    "%s_error" % prefix,
                ]
            )
        return fields

    def lookup(self, target: str, source: str) -> Tuple[Optional[TransformStamped], str]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                rospy.Time(0),
                rospy.Duration(max(self.lookup_timeout, 0.0)),
            )
            return transform, ""
        except Exception as exc:  # pylint: disable=broad-except
            return None, str(exc).replace("\n", " ")[:180]

    def fill_transform(self, row: Dict[str, object], prefix: str, target: str, source: str, now: float) -> None:
        transform, error = self.lookup(target, source)
        if transform is None:
            row["%s_ok" % prefix] = 0
            row["%s_error" % prefix] = error
            self.previous.pop(prefix, None)
            return

        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = quaternion_to_yaw(q)
        stamp = transform.header.stamp.to_sec()
        age = now - stamp if stamp > 0.0 else None

        previous = self.previous.get(prefix)
        xy_step = None
        dyaw = None
        if previous is not None:
            prev_x, prev_y, prev_yaw = previous
            xy_step = math.hypot(t.x - prev_x, t.y - prev_y)
            dyaw = wrap_angle(yaw - prev_yaw)
        self.previous[prefix] = (t.x, t.y, yaw)

        row["%s_ok" % prefix] = 1
        row["%s_stamp" % prefix] = format_float(stamp)
        row["%s_age" % prefix] = format_float(age)
        row["%s_x" % prefix] = format_float(t.x)
        row["%s_y" % prefix] = format_float(t.y)
        row["%s_yaw" % prefix] = format_float(yaw)
        row["%s_yaw_deg" % prefix] = format_float(math.degrees(yaw), "%.3f")
        row["%s_xy_step" % prefix] = format_float(xy_step)
        row["%s_dyaw_deg" % prefix] = format_float(
            math.degrees(dyaw) if dyaw is not None else None,
            "%.3f",
        )
        row["%s_error" % prefix] = ""

    def build_row(self) -> Dict[str, object]:
        now = rospy.Time.now().to_sec()
        row: Dict[str, object] = {
            "time": format_float(now),
            "wall_time": format_float(time.time()),
            "seq": self.seq,
        }
        for prefix, target, source in self.transforms:
            self.fill_transform(row, prefix, target, source, now)
        self.seq += 1
        return row

    def write_row(self) -> None:
        self.writer.writerow(self.build_row())
        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_every_rows:
            self.log_file.flush()
            self.rows_since_flush = 0

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

    def run(self) -> None:
        rate = rospy.Rate(max(self.rate_hz, 0.1))
        while not rospy.is_shutdown():
            self.write_row()
            rate.sleep()


def main() -> None:
    rospy.init_node("tf_diagnostic_logger")
    logger = TfDiagnosticLogger()
    logger.run()


if __name__ == "__main__":
    main()

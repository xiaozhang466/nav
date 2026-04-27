#!/usr/bin/env python3
"""Compose a stable global odometry stream from GPS position and heading.

This node is a pragmatic replacement for the global EKF stage while the system
does not yet have a globally trustworthy heading source. It:

* uses navsat_transform's /odometry/gps x/y as the global position reference
* uses either corrected IMU yaw or local EKF yaw as the robot heading source
* uses the local EKF's twist as the velocity source
* publishes a smooth /odometry/global stream in the configured global frame
* publishes a matching global->odom transform

This keeps the global pose simple and predictable: x/y comes from GPS, yaw comes
from the chosen heading source, and twist comes from local odometry.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


@dataclass
class LocalPoseSample:
    stamp: float
    x: float
    y: float


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class GlobalOdomComposer:
    def __init__(self) -> None:
        self.local_odom_topic = rospy.get_param("~local_odom_topic", "/odometry/filtered")
        self.gps_odom_topic = rospy.get_param("~gps_odom_topic", "/odometry/gps")
        self.output_topic = rospy.get_param("~output_topic", "/odometry/global")
        self.position_source = str(rospy.get_param("~position_source", "gps")).strip().lower()
        self.yaw_source = str(rospy.get_param("~yaw_source", "local")).strip().lower()
        self.imu_topic = rospy.get_param("~imu_topic", "/IMU_data_enu")

        self.map_frame = rospy.get_param("~map_frame", "rtk_map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_link_frame = rospy.get_param("~base_link_frame", "base_link")
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))
        self.zero_altitude = bool(rospy.get_param("~zero_altitude", True))

        self.correction_alpha = float(rospy.get_param("~correction_alpha", 0.25))
        self.correction_alpha = max(0.0, min(1.0, self.correction_alpha))
        self.max_local_interp_gap = float(rospy.get_param("~max_local_interp_gap", 0.2))
        self.gps_timeout = float(rospy.get_param("~gps_timeout", 1.0))
        self.status_interval = float(rospy.get_param("~status_interval", 2.0))

        if self.position_source not in {"gps", "local_plus_xy_correction"}:
            rospy.logwarn(
                "Unknown position_source '%s', falling back to 'gps'",
                self.position_source,
            )
            self.position_source = "gps"
        if self.yaw_source not in {"local", "imu"}:
            rospy.logwarn("Unknown yaw_source '%s', falling back to 'local'", self.yaw_source)
            self.yaw_source = "local"

        self.local_history: deque[LocalPoseSample] = deque(maxlen=2000)
        self.latest_local_msg: Odometry | None = None
        self.latest_gps_msg: Odometry | None = None
        self.latest_gps_covariance: list[float] | None = None
        self.latest_gps_stamp: float | None = None
        self.latest_heading_yaw: float | None = None
        self.latest_heading_covariance: list[float] | None = None
        self.correction_x = 0.0
        self.correction_y = 0.0
        self.has_translation_correction = False

        self.last_status_at = 0.0
        self.last_status_reason = "waiting for GPS and heading"

        self.publisher = rospy.Publisher(self.output_topic, Odometry, queue_size=50)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() if self.publish_tf else None

        rospy.Subscriber(self.local_odom_topic, Odometry, self.local_odom_callback, queue_size=200)
        rospy.Subscriber(self.gps_odom_topic, Odometry, self.gps_odom_callback, queue_size=50)
        if self.yaw_source == "imu":
            rospy.Subscriber(self.imu_topic, Imu, self.imu_callback, queue_size=200)

        rospy.loginfo(
            "global odom composer started: local=%s gps=%s output=%s position_source=%s yaw_source=%s",
            self.local_odom_topic,
            self.gps_odom_topic,
            self.output_topic,
            self.position_source,
            self.yaw_source,
        )

    def maybe_log_status(self) -> None:
        now = rospy.get_time()
        if now - self.last_status_at < self.status_interval:
            return
        self.last_status_at = now

        if self.latest_gps_msg is None or self.latest_heading_yaw is None:
            rospy.loginfo("Global odom pending: %s", self.last_status_reason)
            return
        if self.position_source == "local_plus_xy_correction" and not self.has_translation_correction:
            rospy.loginfo("Global odom pending: %s", self.last_status_reason)
            return

        gps_age = -1.0
        if self.latest_gps_stamp is not None and self.latest_local_msg is not None:
            gps_age = self.latest_local_msg.header.stamp.to_sec() - self.latest_gps_stamp

        rospy.loginfo(
            "Global odom active: position_source=%s yaw_source=%s gps_age=%.3f s",
            self.position_source,
            self.yaw_source,
            gps_age,
        )

    def interpolate_local_pose(self, stamp: float) -> LocalPoseSample | None:
        if not self.local_history:
            return None

        samples = list(self.local_history)
        if len(samples) == 1:
            only = samples[0]
            if abs(stamp - only.stamp) <= self.max_local_interp_gap:
                return only
            return None

        first = samples[0]
        last = samples[-1]
        if stamp <= first.stamp:
            if first.stamp - stamp <= self.max_local_interp_gap:
                return first
            return None
        if stamp >= last.stamp:
            if stamp - last.stamp <= self.max_local_interp_gap:
                return last
            return None

        for index in range(1, len(samples)):
            previous = samples[index - 1]
            current = samples[index]
            if current.stamp < stamp:
                continue

            dt = current.stamp - previous.stamp
            if dt <= 0.0:
                return previous
            if dt > self.max_local_interp_gap:
                return None

            ratio = (stamp - previous.stamp) / dt
            return LocalPoseSample(
                stamp=stamp,
                x=previous.x + ratio * (current.x - previous.x),
                y=previous.y + ratio * (current.y - previous.y),
            )

        return samples[-1]

    def imu_callback(self, msg: Imu) -> None:
        self.latest_heading_yaw = quaternion_to_yaw(msg.orientation)
        self.latest_heading_covariance = list(msg.orientation_covariance)

    def gps_odom_callback(self, msg: Odometry) -> None:
        self.latest_gps_msg = msg
        self.latest_gps_stamp = msg.header.stamp.to_sec()
        self.latest_gps_covariance = list(msg.pose.covariance)
        self.last_status_reason = "received GPS"

        if self.position_source != "local_plus_xy_correction":
            return

        local_pose = self.interpolate_local_pose(self.latest_gps_stamp)
        if local_pose is None:
            self.last_status_reason = "waiting for local odom history near GPS timestamp"
            return

        target_x = msg.pose.pose.position.x - local_pose.x
        target_y = msg.pose.pose.position.y - local_pose.y

        if not self.has_translation_correction:
            self.correction_x = target_x
            self.correction_y = target_y
            self.has_translation_correction = True
            rospy.loginfo(
                "Initialized GPS translation correction: dx=%.3f dy=%.3f",
                self.correction_x,
                self.correction_y,
            )
        else:
            alpha = self.correction_alpha
            self.correction_x += alpha * (target_x - self.correction_x)
            self.correction_y += alpha * (target_y - self.correction_y)
        self.last_status_reason = (
            "latest GPS correction target dx=%.3f dy=%.3f" % (target_x, target_y)
        )

    def local_odom_callback(self, msg: Odometry) -> None:
        self.latest_local_msg = msg
        local_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.local_history.append(
            LocalPoseSample(
                stamp=msg.header.stamp.to_sec(),
                x=msg.pose.pose.position.x,
                y=msg.pose.pose.position.y,
            )
        )
        if self.yaw_source == "local":
            self.latest_heading_yaw = local_yaw

        self.maybe_log_status()

        if self.latest_gps_msg is None:
            self.last_status_reason = "waiting for GPS"
            return

        if self.latest_heading_yaw is None:
            self.last_status_reason = "waiting for heading source"
            return
        if self.position_source == "local_plus_xy_correction" and not self.has_translation_correction:
            self.last_status_reason = "waiting for GPS translation correction"
            return

        if self.latest_gps_stamp is not None:
            gps_age = msg.header.stamp.to_sec() - self.latest_gps_stamp
            if gps_age > self.gps_timeout:
                rospy.logwarn_throttle(
                    2.0,
                    "GPS correction is stale (age %.3f s > %.3f s); publishing last correction.",
                    gps_age,
                    self.gps_timeout,
                )

        global_msg = Odometry()
        global_msg.header.stamp = msg.header.stamp
        global_msg.header.frame_id = self.map_frame
        global_msg.child_frame_id = msg.child_frame_id or self.base_link_frame

        if self.position_source == "gps":
            global_x = self.latest_gps_msg.pose.pose.position.x
            global_y = self.latest_gps_msg.pose.pose.position.y
        else:
            global_x = msg.pose.pose.position.x + self.correction_x
            global_y = msg.pose.pose.position.y + self.correction_y

        global_msg.pose.pose.position.x = global_x
        global_msg.pose.pose.position.y = global_y
        if self.zero_altitude:
            global_msg.pose.pose.position.z = 0.0
        else:
            global_msg.pose.pose.position.z = self.latest_gps_msg.pose.pose.position.z

        global_yaw = self.latest_heading_yaw
        qx, qy, qz, qw = yaw_to_quaternion(global_yaw)
        global_msg.pose.pose.orientation.x = qx
        global_msg.pose.pose.orientation.y = qy
        global_msg.pose.pose.orientation.z = qz
        global_msg.pose.pose.orientation.w = qw

        global_msg.twist = msg.twist

        global_msg.pose.covariance = list(msg.pose.covariance)
        global_msg.twist.covariance = list(msg.twist.covariance)
        if self.latest_gps_covariance is not None and len(self.latest_gps_covariance) == 36:
            global_msg.pose.covariance[0] = self.latest_gps_covariance[0]
            global_msg.pose.covariance[1] = self.latest_gps_covariance[1]
            global_msg.pose.covariance[6] = self.latest_gps_covariance[6]
            global_msg.pose.covariance[7] = self.latest_gps_covariance[7]
        if self.yaw_source == "imu" and self.latest_heading_covariance is not None and len(self.latest_heading_covariance) == 9:
            global_msg.pose.covariance[35] = self.latest_heading_covariance[8]

        self.publisher.publish(global_msg)

        if self.tf_broadcaster is not None:
            yaw_correction = wrap_angle(global_yaw - local_yaw)
            cos_yaw = math.cos(yaw_correction)
            sin_yaw = math.sin(yaw_correction)
            local_x = msg.pose.pose.position.x
            local_y = msg.pose.pose.position.y
            correction_x = global_x - (cos_yaw * local_x - sin_yaw * local_y)
            correction_y = global_y - (sin_yaw * local_x + cos_yaw * local_y)
            transform = TransformStamped()
            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = self.map_frame
            transform.child_frame_id = self.odom_frame
            transform.transform.translation.x = correction_x
            transform.transform.translation.y = correction_y
            transform.transform.translation.z = 0.0
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = math.sin(0.5 * yaw_correction)
            transform.transform.rotation.w = math.cos(0.5 * yaw_correction)
            self.tf_broadcaster.sendTransform(transform)


def main() -> int:
    rospy.init_node("compose_global_odom", anonymous=False)
    GlobalOdomComposer()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

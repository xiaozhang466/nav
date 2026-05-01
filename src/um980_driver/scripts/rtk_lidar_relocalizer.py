#!/usr/bin/env python3
"""Use trusted RTK pose to reset FAST-LOCALIZATION through /initialpose.

This node can run in automatic mode: when RTK quality is trusted and the LiDAR
pose drifts or becomes unhealthy, it republishes /initialpose to pull
FAST-LOCALIZATION back to the RTK pose.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


@dataclass
class Pose2D:
    stamp: float
    x: float
    y: float
    yaw: float


@dataclass
class Pose3D:
    stamp: float
    frame_id: str
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


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


def quaternion_to_rpy(orientation) -> tuple[float, float, float]:
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    yaw = quaternion_to_yaw(orientation)
    return roll, pitch, yaw


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return rpy_to_quaternion(0.0, 0.0, yaw)


def pose_from_odom(msg: Odometry) -> Pose2D:
    pose = msg.pose.pose
    return Pose2D(
        stamp=msg.header.stamp.to_sec(),
        x=pose.position.x,
        y=pose.position.y,
        yaw=quaternion_to_yaw(pose.orientation),
    )


def pose3d_from_odom(msg: Odometry) -> Pose3D:
    pose = msg.pose.pose
    roll, pitch, yaw = quaternion_to_rpy(pose.orientation)
    return Pose3D(
        stamp=msg.header.stamp.to_sec(),
        frame_id=msg.header.frame_id,
        x=pose.position.x,
        y=pose.position.y,
        z=pose.position.z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )


def parse_status_fields(text: str) -> dict[str, str]:
    fields = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


class RtkLidarRelocalizer:
    def __init__(self) -> None:
        self.rtk_odom_topic = rospy.get_param("~rtk_odom_topic", "/odometry/rtk_map")
        self.lidar_odom_topic = rospy.get_param("~lidar_odom_topic", "/odometry/lidar_map")
        self.raw_lidar_odom_topic = rospy.get_param("~raw_lidar_odom_topic", "/Odometry")
        self.lidar_status_topic = rospy.get_param(
            "~lidar_status_topic", "/lidar_localization/status"
        )
        self.rtk_fix_type_topic = rospy.get_param("~rtk_fix_type_topic", "/rtk/fix_type")
        self.rtk_position_type_topic = rospy.get_param(
            "~rtk_position_type_topic", "/rtk/position_type"
        )
        self.initialpose_topic = rospy.get_param("~initialpose_topic", "/initialpose")
        self.status_topic = rospy.get_param(
            "~status_topic", "/rtk_lidar_relocalizer/status"
        )

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.source_timeout = float(rospy.get_param("~source_timeout", 1.0))
        self.status_timeout = float(rospy.get_param("~status_timeout", 2.0))
        self.accepted_fix_types = set(
            rospy.get_param("~accepted_fix_types", ["rtk_fixed"])
        )
        self.accepted_position_types = set(
            rospy.get_param("~accepted_position_types", ["NARROW_INT", "L1_INT"])
        )

        self.yaw_source = str(rospy.get_param("~yaw_source", "lidar")).strip().lower()
        self.allow_rtk_yaw_fallback = bool(
            rospy.get_param("~allow_rtk_yaw_fallback", False)
        )
        self.initialpose_xy_covariance = float(
            rospy.get_param("~initialpose_xy_covariance", 0.04)
        )
        self.initialpose_yaw_covariance = float(
            rospy.get_param("~initialpose_yaw_covariance", 0.03)
        )
        self.publish_count = int(rospy.get_param("~publish_count", 3))
        self.publish_period = float(rospy.get_param("~publish_period", 0.05))
        self.require_healthy_lidar_3d_pose = bool(
            rospy.get_param("~require_healthy_lidar_3d_pose", True)
        )
        self.healthy_lidar_pose_timeout = float(
            rospy.get_param("~healthy_lidar_pose_timeout", 10.0)
        )
        self.lidar_status_stamp_tolerance = float(
            rospy.get_param("~lidar_status_stamp_tolerance", 0.05)
        )
        self.healthy_lidar_max_abs_z = float(
            rospy.get_param("~healthy_lidar_max_abs_z", 20.0)
        )
        self.healthy_lidar_max_abs_roll_pitch = math.radians(
            float(rospy.get_param("~healthy_lidar_max_abs_roll_pitch_deg", 45.0))
        )
        self.healthy_lidar_max_z_step = float(
            rospy.get_param("~healthy_lidar_max_z_step", 0.5)
        )
        self.healthy_lidar_max_roll_pitch_step = math.radians(
            float(rospy.get_param("~healthy_lidar_max_roll_pitch_step_deg", 10.0))
        )

        self.auto_enabled = bool(rospy.get_param("~auto_enabled", False))
        self.auto_check_rate = float(rospy.get_param("~auto_check_rate", 1.0))
        self.auto_min_xy_delta = float(rospy.get_param("~auto_min_xy_delta", 1.0))
        self.auto_min_yaw_delta = math.radians(
            float(rospy.get_param("~auto_min_yaw_delta_deg", 25.0))
        )
        self.auto_min_xy_growth = float(
            rospy.get_param("~auto_min_xy_growth", 0.5)
        )
        self.auto_min_yaw_growth = math.radians(
            float(rospy.get_param("~auto_min_yaw_growth_deg", 8.0))
        )
        self.auto_relocalize_on_lidar_unhealthy = bool(
            rospy.get_param("~auto_relocalize_on_lidar_unhealthy", True)
        )
        self.auto_lidar_bad_reasons = set(
            rospy.get_param(
                "~auto_lidar_bad_reasons",
                [
                    "lidar_jump",
                    "xy_inconsistent",
                    "yaw_inconsistent",
                    "stale_lidar",
                ],
            )
        )
        self.relocalize_cooldown = float(
            rospy.get_param("~relocalize_cooldown", 10.0)
        )
        self.post_relocalize_grace_period = float(
            rospy.get_param("~post_relocalize_grace_period", 15.0)
        )

        self.latest_rtk_msg: Optional[Odometry] = None
        self.latest_lidar_msg: Optional[Odometry] = None
        self.lidar_status_ok: Optional[bool] = None
        self.lidar_status_reason = "waiting_for_lidar_status"
        self.lidar_status_stamp: Optional[float] = None
        self.lidar_status_source_stamp: Optional[float] = None
        self.rtk_fix_type: Optional[str] = None
        self.rtk_fix_type_stamp: Optional[float] = None
        self.rtk_position_type: Optional[str] = None
        self.rtk_position_type_stamp: Optional[float] = None
        self.last_relocalize_time = 0.0
        self.last_status = "ok=false reason=waiting_for_data"
        self.auto_baseline_xy_delta: Optional[float] = None
        self.auto_baseline_yaw_delta: Optional[float] = None
        self.auto_baseline_stamp: Optional[float] = None
        self.raw_lidar_history: deque[Pose3D] = deque(maxlen=300)
        self.latest_healthy_lidar_3d_pose: Optional[Pose3D] = None
        self.last_healthy_lidar_3d_reject_reason = "none"

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.initialpose_pub = rospy.Publisher(
            self.initialpose_topic, PoseWithCovarianceStamped, queue_size=10
        )
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10, latch=True)

        rospy.Subscriber(self.rtk_odom_topic, Odometry, self.rtk_callback, queue_size=20)
        rospy.Subscriber(self.lidar_odom_topic, Odometry, self.lidar_callback, queue_size=20)
        rospy.Subscriber(
            self.raw_lidar_odom_topic, Odometry, self.raw_lidar_callback, queue_size=100
        )
        rospy.Subscriber(
            self.lidar_status_topic, String, self.lidar_status_callback, queue_size=10
        )
        rospy.Subscriber(
            self.rtk_fix_type_topic, String, self.rtk_fix_type_callback, queue_size=10
        )
        rospy.Subscriber(
            self.rtk_position_type_topic,
            String,
            self.rtk_position_type_callback,
            queue_size=10,
        )

        rospy.Service("~relocalize", Trigger, self.handle_relocalize)
        rospy.Timer(rospy.Duration(1.0), self.status_timer_callback)
        if self.auto_enabled:
            rospy.Timer(
                rospy.Duration(1.0 / max(self.auto_check_rate, 0.1)),
                self.auto_timer_callback,
            )

        rospy.loginfo(
            "rtk lidar relocalizer started: rtk=%s lidar=%s raw_lidar=%s lidar_status=%s initialpose=%s yaw_source=%s auto=%s require_healthy_3d=%s",
            self.rtk_odom_topic,
            self.lidar_odom_topic,
            self.raw_lidar_odom_topic,
            self.lidar_status_topic,
            self.initialpose_topic,
            self.yaw_source,
            self.auto_enabled,
            self.require_healthy_lidar_3d_pose,
        )

    def rtk_callback(self, msg: Odometry) -> None:
        self.latest_rtk_msg = msg

    def lidar_callback(self, msg: Odometry) -> None:
        self.latest_lidar_msg = msg

    def raw_lidar_callback(self, msg: Odometry) -> None:
        pose = pose3d_from_odom(msg)
        self.raw_lidar_history.append(pose)
        if (
            self.lidar_status_ok is True
            and self.lidar_status_source_stamp is not None
            and abs(pose.stamp - self.lidar_status_source_stamp)
            <= self.lidar_status_stamp_tolerance
        ):
            self.maybe_update_healthy_lidar_3d_pose(pose)

    def lidar_status_callback(self, msg: String) -> None:
        text = msg.data.strip()
        fields = parse_status_fields(text)
        ok_text = fields.get("ok", "").lower()
        if ok_text == "true":
            self.lidar_status_ok = True
        elif ok_text == "false":
            self.lidar_status_ok = False
        else:
            self.lidar_status_ok = None
        self.lidar_status_reason = fields.get("reason", "unknown_lidar_status")
        self.lidar_status_stamp = rospy.Time.now().to_sec()
        self.lidar_status_source_stamp = None

        source_stamp_text = fields.get("source_stamp")
        if source_stamp_text is not None:
            try:
                self.lidar_status_source_stamp = float(source_stamp_text)
            except ValueError:
                self.lidar_status_source_stamp = None

        if self.lidar_status_ok is True and self.lidar_status_source_stamp is not None:
            self.update_healthy_lidar_3d_from_stamp(self.lidar_status_source_stamp)

    def rtk_fix_type_callback(self, msg: String) -> None:
        self.rtk_fix_type = msg.data.strip()
        self.rtk_fix_type_stamp = rospy.Time.now().to_sec()

    def rtk_position_type_callback(self, msg: String) -> None:
        self.rtk_position_type = msg.data.strip()
        self.rtk_position_type_stamp = rospy.Time.now().to_sec()

    def publish_status(self, text: str) -> None:
        self.last_status = text
        self.status_pub.publish(String(data=text))

    def update_healthy_lidar_3d_from_stamp(self, source_stamp: float) -> None:
        best_pose: Optional[Pose3D] = None
        best_delta: Optional[float] = None
        for pose in self.raw_lidar_history:
            delta = abs(pose.stamp - source_stamp)
            if best_delta is None or delta < best_delta:
                best_pose = pose
                best_delta = delta

        if (
            best_pose is not None
            and best_delta is not None
            and best_delta <= self.lidar_status_stamp_tolerance
        ):
            self.maybe_update_healthy_lidar_3d_pose(best_pose)

    def maybe_update_healthy_lidar_3d_pose(self, pose: Pose3D) -> bool:
        values = (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)
        if not all(math.isfinite(value) for value in values):
            self.last_healthy_lidar_3d_reject_reason = "non_finite_3d_pose"
            return False

        if self.healthy_lidar_max_abs_z > 0.0 and abs(pose.z) > self.healthy_lidar_max_abs_z:
            self.last_healthy_lidar_3d_reject_reason = (
                "abs_z_too_large z=%.3f max=%.3f"
                % (pose.z, self.healthy_lidar_max_abs_z)
            )
            return False

        max_rp = self.healthy_lidar_max_abs_roll_pitch
        if max_rp > 0.0 and (abs(pose.roll) > max_rp or abs(pose.pitch) > max_rp):
            self.last_healthy_lidar_3d_reject_reason = (
                "roll_pitch_too_large roll_deg=%.2f pitch_deg=%.2f max_deg=%.2f"
                % (math.degrees(pose.roll), math.degrees(pose.pitch), math.degrees(max_rp))
            )
            return False

        previous = self.latest_healthy_lidar_3d_pose
        if previous is not None and pose.stamp >= previous.stamp:
            z_step = abs(pose.z - previous.z)
            if self.healthy_lidar_max_z_step > 0.0 and z_step > self.healthy_lidar_max_z_step:
                self.last_healthy_lidar_3d_reject_reason = (
                    "z_step_too_large step=%.3f max=%.3f"
                    % (z_step, self.healthy_lidar_max_z_step)
                )
                return False

            roll_step = abs(wrap_angle(pose.roll - previous.roll))
            pitch_step = abs(wrap_angle(pose.pitch - previous.pitch))
            max_rp_step = self.healthy_lidar_max_roll_pitch_step
            if max_rp_step > 0.0 and (
                roll_step > max_rp_step or pitch_step > max_rp_step
            ):
                self.last_healthy_lidar_3d_reject_reason = (
                    "roll_pitch_step_too_large roll_step_deg=%.2f pitch_step_deg=%.2f max_deg=%.2f"
                    % (
                        math.degrees(roll_step),
                        math.degrees(pitch_step),
                        math.degrees(max_rp_step),
                    )
                )
                return False

        self.latest_healthy_lidar_3d_pose = pose
        self.last_healthy_lidar_3d_reject_reason = "none"
        return True

    def healthy_lidar_3d_pose_for_reset(
        self, now: float
    ) -> tuple[Optional[Pose3D], str]:
        if self.latest_healthy_lidar_3d_pose is None:
            return None, "no_healthy_lidar_3d_pose"

        age = now - self.latest_healthy_lidar_3d_pose.stamp
        if self.healthy_lidar_pose_timeout > 0.0 and age > self.healthy_lidar_pose_timeout:
            return (
                None,
                "stale_healthy_lidar_3d_pose_age=%.3f timeout=%.3f"
                % (age, self.healthy_lidar_pose_timeout),
            )

        return self.latest_healthy_lidar_3d_pose, "ok"

    def reset_auto_drift_baseline(self) -> None:
        self.auto_baseline_xy_delta = None
        self.auto_baseline_yaw_delta = None
        self.auto_baseline_stamp = None

    def update_auto_drift_baseline(
        self, xy_delta: float, yaw_delta: float, now: float
    ) -> tuple[float, float]:
        if self.auto_baseline_xy_delta is None or self.auto_baseline_yaw_delta is None:
            self.auto_baseline_xy_delta = xy_delta
            self.auto_baseline_yaw_delta = yaw_delta
        else:
            # Keep the best agreement seen since the last reset so a stable RTK/LiDAR
            # offset does not retrigger relocalization every cooldown period.
            self.auto_baseline_xy_delta = min(self.auto_baseline_xy_delta, xy_delta)
            self.auto_baseline_yaw_delta = min(self.auto_baseline_yaw_delta, yaw_delta)
        self.auto_baseline_stamp = now
        return self.auto_baseline_xy_delta, self.auto_baseline_yaw_delta

    def auto_drift_trigger_reason(
        self, xy_delta: float, yaw_delta: float, now: float
    ) -> Optional[str]:
        baseline_xy, baseline_yaw = self.update_auto_drift_baseline(
            xy_delta, yaw_delta, now
        )
        xy_growth = max(0.0, xy_delta - baseline_xy)
        yaw_growth = max(0.0, yaw_delta - baseline_yaw)

        xy_trigger = (
            xy_delta >= self.auto_min_xy_delta and xy_growth >= self.auto_min_xy_growth
        )
        yaw_trigger = (
            yaw_delta >= self.auto_min_yaw_delta and yaw_growth >= self.auto_min_yaw_growth
        )
        if not xy_trigger and not yaw_trigger:
            return None

        return (
            "baseline_xy=%.3f baseline_yaw_deg=%.2f "
            "xy_growth=%.3f yaw_growth_deg=%.2f"
            % (
                baseline_xy,
                math.degrees(baseline_yaw),
                xy_growth,
                math.degrees(yaw_growth),
            )
        )

    def latest_pose_is_fresh(self, pose: Pose2D, now: float) -> bool:
        return now - pose.stamp <= self.source_timeout

    def rtk_is_usable(self) -> tuple[bool, str]:
        now = rospy.Time.now().to_sec()
        if self.rtk_fix_type is None or self.rtk_position_type is None:
            return False, "waiting_for_rtk_status"
        if (
            self.rtk_fix_type_stamp is not None
            and now - self.rtk_fix_type_stamp > self.status_timeout
        ):
            return False, "stale_rtk_fix_type"
        if (
            self.rtk_position_type_stamp is not None
            and now - self.rtk_position_type_stamp > self.status_timeout
        ):
            return False, "stale_rtk_position_type"
        if self.rtk_fix_type not in self.accepted_fix_types:
            return False, "bad_rtk_fix_type:%s" % self.rtk_fix_type
        if self.rtk_position_type not in self.accepted_position_types:
            return False, "bad_rtk_position_type:%s" % self.rtk_position_type
        return True, "ok"

    def transform_pose_to_map(self, msg: Odometry) -> tuple[Optional[Pose2D], str]:
        source_frame = msg.header.frame_id or self.map_frame
        pose = pose_from_odom(msg)
        if source_frame == self.map_frame:
            return pose, "ok"

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.1),
            )
        except Exception as exc:  # pylint: disable=broad-except
            return None, "tf_lookup_failed:%s_to_%s:%s" % (
                source_frame,
                self.map_frame,
                exc,
            )

        t = transform.transform.translation
        q = transform.transform.rotation
        tf_yaw = quaternion_to_yaw(q)
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        return (
            Pose2D(
                stamp=pose.stamp,
                x=t.x + cos_yaw * pose.x - sin_yaw * pose.y,
                y=t.y + sin_yaw * pose.x + cos_yaw * pose.y,
                yaw=wrap_angle(tf_yaw + pose.yaw),
            ),
            "ok",
        )

    def choose_reset_yaw(
        self, rtk_pose: Pose2D, lidar_pose: Optional[Pose2D]
    ) -> tuple[Optional[float], str]:
        if self.yaw_source == "rtk":
            return rtk_pose.yaw, "rtk"
        if self.yaw_source == "lidar":
            if lidar_pose is not None:
                return lidar_pose.yaw, "lidar"
            if self.allow_rtk_yaw_fallback:
                return rtk_pose.yaw, "rtk_fallback"
            return None, "no_fresh_lidar_yaw"
        return None, "bad_yaw_source:%s" % self.yaw_source

    def make_initialpose_msg(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0] = self.initialpose_xy_covariance
        msg.pose.covariance[7] = self.initialpose_xy_covariance
        msg.pose.covariance[35] = self.initialpose_yaw_covariance
        return msg

    def current_delta_text(
        self, rtk_pose: Pose2D, lidar_pose: Optional[Pose2D]
    ) -> tuple[float, float, str]:
        if lidar_pose is None:
            return float("nan"), float("nan"), "delta=unavailable"
        xy_delta = math.hypot(rtk_pose.x - lidar_pose.x, rtk_pose.y - lidar_pose.y)
        yaw_delta = wrap_angle(rtk_pose.yaw - lidar_pose.yaw)
        return (
            xy_delta,
            yaw_delta,
            "delta_xy=%.3f delta_yaw_deg=%.2f" % (xy_delta, math.degrees(yaw_delta)),
        )

    def latest_lidar_pose_in_map(self, now: float) -> tuple[Optional[Pose2D], str, bool]:
        if self.latest_lidar_msg is None:
            return None, "waiting_for_lidar_odom", False

        lidar_pose, lidar_reason = self.transform_pose_to_map(self.latest_lidar_msg)
        if lidar_pose is None:
            return None, lidar_reason, False
        if not self.latest_pose_is_fresh(lidar_pose, now):
            return lidar_pose, "stale_lidar_odom_age=%.3f" % (now - lidar_pose.stamp), False
        return lidar_pose, "ok", True

    def lidar_unhealthy_auto_reason(
        self, lidar_reason: str, lidar_fresh: bool, now: float
    ) -> Optional[str]:
        if not self.auto_relocalize_on_lidar_unhealthy:
            return None
        if self.yaw_source != "rtk":
            return None

        if self.lidar_status_stamp is not None:
            status_age = now - self.lidar_status_stamp
            if status_age > self.status_timeout:
                return "stale_lidar_status_age=%.3f" % status_age
            if (
                self.lidar_status_ok is False
                and self.lidar_status_reason in self.auto_lidar_bad_reasons
            ):
                return "bad_lidar_status:%s" % self.lidar_status_reason

        if self.latest_lidar_msg is not None and not lidar_fresh:
            return lidar_reason

        return None

    def build_and_publish_initialpose(self, reason: str) -> tuple[bool, str]:
        now = rospy.Time.now().to_sec()
        if self.latest_rtk_msg is None:
            return False, "waiting_for_rtk_odom"

        rtk_ok, rtk_reason = self.rtk_is_usable()
        if not rtk_ok:
            return False, rtk_reason

        rtk_pose, transform_reason = self.transform_pose_to_map(self.latest_rtk_msg)
        if rtk_pose is None:
            return False, transform_reason
        if not self.latest_pose_is_fresh(rtk_pose, now):
            return False, "stale_rtk_odom_age=%.3f" % (now - rtk_pose.stamp)

        lidar_pose, lidar_reason, lidar_fresh = self.latest_lidar_pose_in_map(now)
        if not lidar_fresh and self.yaw_source == "lidar":
            return False, "lidar_yaw_unavailable:%s" % lidar_reason

        yaw, yaw_reason = self.choose_reset_yaw(rtk_pose, lidar_pose)
        if yaw is None:
            return False, yaw_reason

        healthy_3d_pose, healthy_3d_reason = self.healthy_lidar_3d_pose_for_reset(now)
        if healthy_3d_pose is None:
            if self.require_healthy_lidar_3d_pose:
                return False, healthy_3d_reason
            reset_z = 0.0
            reset_roll = 0.0
            reset_pitch = 0.0
        else:
            reset_z = healthy_3d_pose.z
            reset_roll = healthy_3d_pose.roll
            reset_pitch = healthy_3d_pose.pitch

        msg = self.make_initialpose_msg(
            rtk_pose.x, rtk_pose.y, reset_z, reset_roll, reset_pitch, yaw
        )
        for index in range(max(1, self.publish_count)):
            self.initialpose_pub.publish(msg)
            if index + 1 < self.publish_count and self.publish_period > 0.0:
                rospy.sleep(self.publish_period)

        self.last_relocalize_time = now
        self.reset_auto_drift_baseline()
        xy_delta, yaw_delta, delta_text = self.current_delta_text(rtk_pose, lidar_pose)
        healthy_3d_age = (
            now - healthy_3d_pose.stamp if healthy_3d_pose is not None else float("nan")
        )
        status = (
            "ok=true reason=%s yaw_source=%s x=%.3f y=%.3f z=%.3f "
            "roll_deg=%.2f pitch_deg=%.2f yaw_deg=%.2f "
            "lidar_3d_age=%.3f rtk_fix=%s rtk_position=%s %s"
            % (
                reason,
                yaw_reason,
                rtk_pose.x,
                rtk_pose.y,
                reset_z,
                math.degrees(reset_roll),
                math.degrees(reset_pitch),
                math.degrees(yaw),
                healthy_3d_age,
                self.rtk_fix_type,
                self.rtk_position_type,
                delta_text,
            )
        )
        self.publish_status(status)
        return (
            True,
            "%s published /initialpose; %s xy=%.3f yaw_deg=%.2f"
            % (reason, yaw_reason, xy_delta, math.degrees(yaw_delta)),
        )

    def handle_relocalize(self, _request) -> TriggerResponse:
        ok, message = self.build_and_publish_initialpose("manual_trigger")
        if not ok:
            self.publish_status("ok=false reason=%s" % message)
        return TriggerResponse(success=ok, message=message)

    def auto_timer_callback(self, _event) -> None:
        now = rospy.Time.now().to_sec()
        if now - self.last_relocalize_time < self.relocalize_cooldown:
            return
        if (
            self.last_relocalize_time > 0.0
            and now - self.last_relocalize_time < self.post_relocalize_grace_period
        ):
            return
        if self.latest_rtk_msg is None:
            return

        rtk_pose, rtk_reason = self.transform_pose_to_map(self.latest_rtk_msg)
        if rtk_pose is None:
            self.publish_status("ok=false reason=auto_tf_failed:%s" % rtk_reason)
            return
        if not self.latest_pose_is_fresh(rtk_pose, now):
            self.publish_status("ok=false reason=auto_stale_rtk_odom_age=%.3f" % (now - rtk_pose.stamp))
            return

        rtk_ok, rtk_reason = self.rtk_is_usable()
        if not rtk_ok:
            return

        lidar_pose, lidar_reason, lidar_fresh = self.latest_lidar_pose_in_map(now)
        if lidar_fresh and lidar_pose is not None:
            xy_delta = math.hypot(rtk_pose.x - lidar_pose.x, rtk_pose.y - lidar_pose.y)
            yaw_delta = abs(wrap_angle(rtk_pose.yaw - lidar_pose.yaw))
            drift_reason = self.auto_drift_trigger_reason(xy_delta, yaw_delta, now)
            if drift_reason is None:
                return
            reason = "auto_drift:%s" % drift_reason
        else:
            unhealthy_reason = self.lidar_unhealthy_auto_reason(lidar_reason, lidar_fresh, now)
            if unhealthy_reason is None:
                return
            reason = "auto_lidar_unhealthy:%s" % unhealthy_reason

        ok, message = self.build_and_publish_initialpose(reason)
        if not ok:
            self.publish_status("ok=false reason=auto_failed:%s" % message)

    def lidar_status_text(self, now: float) -> str:
        if self.lidar_status_stamp is None:
            return "lidar_status=none lidar_status_reason=waiting_for_lidar_status"
        age = now - self.lidar_status_stamp
        return (
            "lidar_status_ok=%s lidar_status_reason=%s lidar_status_age=%.2f"
            % (
                str(self.lidar_status_ok).lower(),
                self.lidar_status_reason,
                age,
            )
        )

    def auto_baseline_text(self) -> str:
        if self.auto_baseline_xy_delta is None or self.auto_baseline_yaw_delta is None:
            return "auto_baseline=none"
        return (
            "auto_baseline_xy=%.3f auto_baseline_yaw_deg=%.2f"
            % (
                self.auto_baseline_xy_delta,
                math.degrees(self.auto_baseline_yaw_delta),
            )
        )

    def healthy_lidar_3d_text(self, now: float) -> str:
        pose = self.latest_healthy_lidar_3d_pose
        if pose is None:
            return "healthy_lidar_3d=none healthy_lidar_3d_reject=%s" % (
                self.last_healthy_lidar_3d_reject_reason
            )
        return (
            "healthy_lidar_3d_age=%.3f healthy_lidar_z=%.3f "
            "healthy_lidar_roll_deg=%.2f healthy_lidar_pitch_deg=%.2f "
            "healthy_lidar_3d_reject=%s"
            % (
                now - pose.stamp,
                pose.z,
                math.degrees(pose.roll),
                math.degrees(pose.pitch),
                self.last_healthy_lidar_3d_reject_reason,
            )
        )

    def status_timer_callback(self, _event) -> None:
        now = rospy.Time.now().to_sec()
        if self.latest_rtk_msg is None:
            self.status_pub.publish(String(data=self.last_status))
            return

        rtk_pose, rtk_reason = self.transform_pose_to_map(self.latest_rtk_msg)
        lidar_pose = None
        if self.latest_lidar_msg is not None:
            lidar_pose, _ = self.transform_pose_to_map(self.latest_lidar_msg)

        rtk_ok, rtk_quality_reason = self.rtk_is_usable()
        if rtk_pose is None:
            status = "ok=false reason=%s %s" % (rtk_reason, self.lidar_status_text(now))
        else:
            _, _, delta_text = self.current_delta_text(rtk_pose, lidar_pose)
            status = (
                "ok=%s reason=%s rtk_fix=%s rtk_position=%s yaw_source=%s %s %s %s %s"
                % (
                    str(rtk_ok).lower(),
                    rtk_quality_reason,
                    self.rtk_fix_type,
                    self.rtk_position_type,
                    self.yaw_source,
                    delta_text,
                    self.auto_baseline_text(),
                    self.healthy_lidar_3d_text(now),
                    self.lidar_status_text(now),
                )
            )
        self.status_pub.publish(String(data=status))


if __name__ == "__main__":
    rospy.init_node("rtk_lidar_relocalizer")
    RtkLidarRelocalizer()
    rospy.spin()

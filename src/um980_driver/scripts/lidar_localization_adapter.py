#!/usr/bin/env python3
"""Normalize FAST-LOCALIZATION odometry for later RTK/LiDAR selection.

This node intentionally does not publish TF by default. It converts the FAST
`camera_init -> body` odometry naming into a project-level odometry topic:
`map -> base_link`, and publishes a lightweight status string based on short
term consistency against the local wheel+IMU odometry.
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


@dataclass
class Pose2D:
    stamp: float
    x: float
    y: float
    yaw: float


@dataclass
class Delta2D:
    dx: float
    dy: float
    dyaw: float


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


def yaw_to_quaternion(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def pose_from_odom(msg: Odometry) -> Pose2D:
    pose = msg.pose.pose
    return Pose2D(
        stamp=msg.header.stamp.to_sec(),
        x=pose.position.x,
        y=pose.position.y,
        yaw=quaternion_to_yaw(pose.orientation),
    )


def relative_delta(previous: Pose2D, current: Pose2D) -> Delta2D:
    dx_world = current.x - previous.x
    dy_world = current.y - previous.y
    cos_yaw = math.cos(previous.yaw)
    sin_yaw = math.sin(previous.yaw)
    return Delta2D(
        dx=cos_yaw * dx_world + sin_yaw * dy_world,
        dy=-sin_yaw * dx_world + cos_yaw * dy_world,
        dyaw=wrap_angle(current.yaw - previous.yaw),
    )


def has_nonzero_covariance(covariance) -> bool:
    return any(abs(value) > 1e-12 for value in covariance)


class LidarLocalizationAdapter:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/Odometry")
        self.local_odom_topic = rospy.get_param("~local_odom_topic", "/odometry/filtered")
        self.output_topic = rospy.get_param("~output_topic", "/odometry/lidar_map")
        self.status_topic = rospy.get_param("~status_topic", "/lidar_localization/status")

        self.input_frame = rospy.get_param("~input_frame", "camera_init")
        self.input_child_frame = rospy.get_param("~input_child_frame", "body")
        self.output_frame = rospy.get_param("~output_frame", "map")
        self.output_child_frame = rospy.get_param("~output_child_frame", "base_link")
        self.assume_body_is_base_link = bool(
            rospy.get_param("~assume_body_is_base_link", True)
        )
        self.strict_input_frame = bool(rospy.get_param("~strict_input_frame", True))
        self.flatten_to_2d = bool(rospy.get_param("~flatten_to_2d", True))

        self.nominal_xy_covariance = float(rospy.get_param("~nominal_xy_covariance", 0.0025))
        self.nominal_yaw_covariance = float(rospy.get_param("~nominal_yaw_covariance", 0.0004))
        self.bad_xy_covariance = float(rospy.get_param("~bad_xy_covariance", 0.25))
        self.bad_yaw_covariance = float(rospy.get_param("~bad_yaw_covariance", 0.04))

        self.max_consistency_xy_error = float(
            rospy.get_param("~max_consistency_xy_error", 0.20)
        )
        self.max_consistency_yaw_error = math.radians(
            float(rospy.get_param("~max_consistency_yaw_error_deg", 12.0))
        )
        self.max_lidar_step = float(rospy.get_param("~max_lidar_step", 1.0))
        self.max_lidar_yaw_step = math.radians(
            float(rospy.get_param("~max_lidar_yaw_step_deg", 35.0))
        )
        self.recovery_enabled = bool(rospy.get_param("~recovery_enabled", True))
        self.recovery_required_samples = int(
            rospy.get_param("~recovery_required_samples", 8)
        )
        self.recovery_max_consistency_xy_error = float(
            rospy.get_param("~recovery_max_consistency_xy_error", 0.25)
        )
        self.recovery_max_consistency_yaw_error = math.radians(
            float(rospy.get_param("~recovery_max_consistency_yaw_error_deg", 15.0))
        )
        self.recovery_max_lidar_step = float(
            rospy.get_param("~recovery_max_lidar_step", 0.6)
        )
        self.recovery_max_lidar_yaw_step = math.radians(
            float(rospy.get_param("~recovery_max_lidar_yaw_step_deg", 30.0))
        )
        self.recovery_min_quality = float(rospy.get_param("~recovery_min_quality", 0.5))
        self.recovery_reasons = set(
            rospy.get_param(
                "~recovery_reasons",
                ["lidar_jump", "xy_inconsistent", "yaw_inconsistent"],
            )
        )
        self.max_abs_pose_xy = float(rospy.get_param("~max_abs_pose_xy", 500.0))
        self.max_abs_input_z = float(rospy.get_param("~max_abs_input_z", 50.0))
        self.stale_timeout = float(rospy.get_param("~stale_timeout", 0.5))
        self.status_interval = float(rospy.get_param("~status_interval", 1.0))

        self.latest_local_pose: Pose2D | None = None
        self.previous_lidar_pose: Pose2D | None = None
        self.previous_local_at_lidar: Pose2D | None = None
        self.last_lidar_stamp: float | None = None
        self.last_status = "ok=false quality=0.000 reason=waiting_for_lidar"
        self.recovery_lidar_pose: Pose2D | None = None
        self.recovery_local_pose: Pose2D | None = None
        self.recovery_count = 0

        self.odom_pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10, latch=True)

        rospy.Subscriber(self.local_odom_topic, Odometry, self.local_odom_callback, queue_size=100)
        rospy.Subscriber(self.input_topic, Odometry, self.lidar_odom_callback, queue_size=50)
        rospy.Timer(rospy.Duration(self.status_interval), self.timer_callback)

        rospy.loginfo(
            "lidar localization adapter started: input=%s (%s->%s) output=%s (%s->%s)",
            self.input_topic,
            self.input_frame,
            self.input_child_frame,
            self.output_topic,
            self.output_frame,
            self.output_child_frame,
        )

    def local_odom_callback(self, msg: Odometry) -> None:
        self.latest_local_pose = pose_from_odom(msg)

    def child_frame_is_usable(self, child_frame: str) -> bool:
        if child_frame == self.output_child_frame:
            return True
        if (
            self.assume_body_is_base_link
            and child_frame == self.input_child_frame
            and self.output_child_frame == "base_link"
        ):
            return True
        return False

    def pose_is_in_bounds(self, pose: Pose2D, input_z: float) -> tuple[bool, str]:
        if self.max_abs_pose_xy > 0.0 and (
            abs(pose.x) > self.max_abs_pose_xy or abs(pose.y) > self.max_abs_pose_xy
        ):
            return (
                False,
                "pose_out_of_bounds x=%.3f y=%.3f max_abs_xy=%.3f"
                % (pose.x, pose.y, self.max_abs_pose_xy),
            )

        if self.max_abs_input_z > 0.0 and abs(input_z) > self.max_abs_input_z:
            return (
                False,
                "pose_z_out_of_bounds z=%.3f max_abs_z=%.3f"
                % (input_z, self.max_abs_input_z),
            )

        return True, "ok"

    def make_output_msg(self, msg: Odometry, ok: bool) -> Odometry:
        out = Odometry()
        out.header = copy.deepcopy(msg.header)
        out.header.frame_id = self.output_frame
        out.child_frame_id = self.output_child_frame
        out.pose = copy.deepcopy(msg.pose)
        out.twist = copy.deepcopy(msg.twist)

        if self.flatten_to_2d:
            yaw = quaternion_to_yaw(out.pose.pose.orientation)
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = qx
            out.pose.pose.orientation.y = qy
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

        if not ok or not has_nonzero_covariance(out.pose.covariance):
            xy_cov = self.nominal_xy_covariance if ok else self.bad_xy_covariance
            yaw_cov = self.nominal_yaw_covariance if ok else self.bad_yaw_covariance
            out.pose.covariance = list(out.pose.covariance)
            out.pose.covariance[0] = xy_cov
            out.pose.covariance[7] = xy_cov
            out.pose.covariance[35] = yaw_cov

        return out

    def score_quality(
        self,
        xy_error: float,
        yaw_error: float,
        lidar_step: float,
        lidar_yaw_step: float,
        max_xy_error: float | None = None,
        max_yaw_error: float | None = None,
        max_lidar_step: float | None = None,
        max_lidar_yaw_step: float | None = None,
    ) -> float:
        max_xy_error = self.max_consistency_xy_error if max_xy_error is None else max_xy_error
        max_yaw_error = (
            self.max_consistency_yaw_error if max_yaw_error is None else max_yaw_error
        )
        max_lidar_step = self.max_lidar_step if max_lidar_step is None else max_lidar_step
        max_lidar_yaw_step = (
            self.max_lidar_yaw_step
            if max_lidar_yaw_step is None
            else max_lidar_yaw_step
        )
        xy_score = 1.0 - min(1.0, xy_error / max(max_xy_error, 1e-6))
        yaw_score = 1.0 - min(1.0, yaw_error / max(max_yaw_error, 1e-6))
        step_score = 1.0 - min(
            1.0,
            max(0.0, lidar_step - max_lidar_step) / max(max_lidar_step, 1e-6),
        )
        yaw_step_score = 1.0 - min(
            1.0,
            max(0.0, lidar_yaw_step - max_lidar_yaw_step)
            / max(max_lidar_yaw_step, 1e-6),
        )
        return max(0.0, min(xy_score, yaw_score, step_score, yaw_step_score))

    def update_status(self, status: str) -> None:
        self.last_status = status
        self.status_pub.publish(String(data=status))

    def format_status(
        self,
        ok: bool,
        quality: float,
        reason: str,
        source_stamp: float,
        xy_error: float = 0.0,
        yaw_error: float = 0.0,
        lidar_step: float = 0.0,
        lidar_yaw_step: float = 0.0,
        extra: str = "",
    ) -> str:
        status = (
            "ok=%s quality=%.3f reason=%s source_stamp=%.9f "
            "xy_error=%.3f yaw_error_deg=%.2f "
            "lidar_step=%.3f lidar_yaw_step_deg=%.2f"
            % (
                str(ok).lower(),
                quality,
                reason,
                source_stamp,
                xy_error,
                math.degrees(yaw_error),
                lidar_step,
                math.degrees(lidar_yaw_step),
            )
        )
        if extra:
            status = "%s %s" % (status, extra)
        return status

    def reset_recovery(self) -> None:
        self.recovery_lidar_pose = None
        self.recovery_local_pose = None
        self.recovery_count = 0

    def recovery_quality(
        self,
        xy_error: float,
        yaw_error: float,
        lidar_step: float,
        lidar_yaw_step: float,
    ) -> float:
        return self.score_quality(
            xy_error,
            yaw_error,
            lidar_step,
            lidar_yaw_step,
            self.recovery_max_consistency_xy_error,
            self.recovery_max_consistency_yaw_error,
            self.recovery_max_lidar_step,
            self.recovery_max_lidar_yaw_step,
        )

    def update_recovery_state(
        self,
        lidar_pose: Pose2D,
        local_pose: Pose2D | None,
        trigger_reason: str,
    ) -> tuple[bool, float, str, float, float, float, float, str]:
        if (
            not self.recovery_enabled
            or trigger_reason not in self.recovery_reasons
            or local_pose is None
        ):
            self.reset_recovery()
            return False, 0.0, trigger_reason, 0.0, 0.0, 0.0, 0.0, ""

        required_samples = max(1, self.recovery_required_samples)
        if self.recovery_lidar_pose is None or self.recovery_local_pose is None:
            self.recovery_lidar_pose = lidar_pose
            self.recovery_local_pose = local_pose
            self.recovery_count = 1
            recovered = self.recovery_count >= required_samples
            extra = "trigger=%s recovery_count=%d recovery_required=%d" % (
                trigger_reason,
                self.recovery_count,
                required_samples,
            )
            if recovered:
                self.reset_recovery()
                return True, 0.5, "relocalized", 0.0, 0.0, 0.0, 0.0, extra
            return False, 0.0, "relocalizing", 0.0, 0.0, 0.0, 0.0, extra

        lidar_delta = relative_delta(self.recovery_lidar_pose, lidar_pose)
        local_delta = relative_delta(self.recovery_local_pose, local_pose)
        xy_error = math.hypot(lidar_delta.dx - local_delta.dx, lidar_delta.dy - local_delta.dy)
        yaw_error = abs(wrap_angle(lidar_delta.dyaw - local_delta.dyaw))
        lidar_step = math.hypot(lidar_delta.dx, lidar_delta.dy)
        lidar_yaw_step = abs(lidar_delta.dyaw)
        quality = self.recovery_quality(
            xy_error, yaw_error, lidar_step, lidar_yaw_step
        )
        stable = (
            xy_error <= self.recovery_max_consistency_xy_error
            and yaw_error <= self.recovery_max_consistency_yaw_error
            and lidar_step <= self.recovery_max_lidar_step
            and lidar_yaw_step <= self.recovery_max_lidar_yaw_step
            and quality >= self.recovery_min_quality
        )

        if stable:
            self.recovery_count += 1
            self.recovery_lidar_pose = lidar_pose
            self.recovery_local_pose = local_pose
            extra = "trigger=%s recovery_count=%d recovery_required=%d" % (
                trigger_reason,
                self.recovery_count,
                required_samples,
            )
            if self.recovery_count >= required_samples:
                self.reset_recovery()
                return (
                    True,
                    max(self.recovery_min_quality, quality),
                    "relocalized",
                    xy_error,
                    yaw_error,
                    lidar_step,
                    lidar_yaw_step,
                    extra,
                )
            return (
                False,
                quality,
                "relocalizing",
                xy_error,
                yaw_error,
                lidar_step,
                lidar_yaw_step,
                extra,
            )

        self.recovery_lidar_pose = lidar_pose
        self.recovery_local_pose = local_pose
        self.recovery_count = 1
        extra = "trigger=%s recovery_count=1 recovery_required=%d" % (
            trigger_reason,
            required_samples,
        )
        return (
            False,
            quality,
            "relocalizing_reset",
            xy_error,
            yaw_error,
            lidar_step,
            lidar_yaw_step,
            extra,
        )

    def lidar_odom_callback(self, msg: Odometry) -> None:
        if self.strict_input_frame and msg.header.frame_id and msg.header.frame_id != self.input_frame:
            self.reset_recovery()
            self.update_status(
                "ok=false quality=0.000 reason=unexpected_frame source_stamp=%.9f frame=%s expected=%s"
                % (msg.header.stamp.to_sec(), msg.header.frame_id, self.input_frame)
            )
            rospy.logwarn_throttle(
                2.0,
                "Dropping LiDAR odom with frame '%s'; expected '%s'",
                msg.header.frame_id,
                self.input_frame,
            )
            return

        if not self.child_frame_is_usable(msg.child_frame_id):
            self.reset_recovery()
            self.update_status(
                "ok=false quality=0.000 reason=unexpected_child source_stamp=%.9f child=%s expected=%s"
                % (msg.header.stamp.to_sec(), msg.child_frame_id, self.input_child_frame)
            )
            rospy.logwarn_throttle(
                2.0,
                "Dropping LiDAR odom with child '%s'; expected '%s' or '%s'",
                msg.child_frame_id,
                self.input_child_frame,
                self.output_child_frame,
            )
            return

        lidar_pose = pose_from_odom(msg)
        self.last_lidar_stamp = lidar_pose.stamp
        in_bounds, bounds_reason = self.pose_is_in_bounds(
            lidar_pose, msg.pose.pose.position.z
        )
        if not in_bounds:
            self.reset_recovery()
            self.update_status(
                self.format_status(False, 0.0, bounds_reason, lidar_pose.stamp)
            )
            return

        ok = False
        quality = 0.0
        reason = "initializing"
        xy_error = 0.0
        yaw_error = 0.0
        lidar_step = 0.0
        lidar_yaw_step = 0.0

        local_pose = self.latest_local_pose
        if local_pose is None:
            reason = "waiting_for_local_odom"
        elif self.previous_lidar_pose is None or self.previous_local_at_lidar is None:
            ok = True
            quality = 0.5
        else:
            lidar_delta = relative_delta(self.previous_lidar_pose, lidar_pose)
            local_delta = relative_delta(self.previous_local_at_lidar, local_pose)
            xy_error = math.hypot(lidar_delta.dx - local_delta.dx, lidar_delta.dy - local_delta.dy)
            yaw_error = abs(wrap_angle(lidar_delta.dyaw - local_delta.dyaw))
            lidar_step = math.hypot(lidar_delta.dx, lidar_delta.dy)
            lidar_yaw_step = abs(lidar_delta.dyaw)

            quality = self.score_quality(xy_error, yaw_error, lidar_step, lidar_yaw_step)
            if lidar_step > self.max_lidar_step or lidar_yaw_step > self.max_lidar_yaw_step:
                reason = "lidar_jump"
            elif xy_error > self.max_consistency_xy_error:
                reason = "xy_inconsistent"
            elif yaw_error > self.max_consistency_yaw_error:
                reason = "yaw_inconsistent"
            else:
                ok = True
                reason = "ok"

        status_extra = ""
        if ok:
            self.reset_recovery()
        else:
            (
                ok,
                recovery_quality,
                recovery_reason,
                recovery_xy_error,
                recovery_yaw_error,
                recovery_lidar_step,
                recovery_lidar_yaw_step,
                status_extra,
            ) = self.update_recovery_state(lidar_pose, local_pose, reason)
            if recovery_reason != reason:
                quality = recovery_quality
                reason = recovery_reason
                xy_error = recovery_xy_error
                yaw_error = recovery_yaw_error
                lidar_step = recovery_lidar_step
                lidar_yaw_step = recovery_lidar_yaw_step

        self.update_status(
            self.format_status(
                ok,
                quality,
                reason,
                lidar_pose.stamp,
                xy_error,
                yaw_error,
                lidar_step,
                lidar_yaw_step,
                status_extra,
            )
        )

        if not ok:
            return

        self.odom_pub.publish(self.make_output_msg(msg, True))

        if local_pose is not None:
            self.previous_local_at_lidar = local_pose
        self.previous_lidar_pose = lidar_pose

    def timer_callback(self, _event) -> None:
        if self.last_lidar_stamp is None:
            self.status_pub.publish(String(data=self.last_status))
            return

        age = rospy.Time.now().to_sec() - self.last_lidar_stamp
        if age > self.stale_timeout:
            self.update_status(
                "ok=false quality=0.000 reason=stale_lidar age=%.3f timeout=%.3f"
                % (age, self.stale_timeout)
            )


if __name__ == "__main__":
    rospy.init_node("lidar_localization_adapter")
    LidarLocalizationAdapter()
    rospy.spin()

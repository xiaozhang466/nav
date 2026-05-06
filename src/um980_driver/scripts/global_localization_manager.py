#!/usr/bin/env python3
"""Select the active global localization source and publish map -> odom.

Inputs are global robot poses from RTK and LiDAR localization. The local EKF
still owns the continuous odom -> base_link transform; this manager owns the
single global correction transform map -> odom.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


@dataclass
class Pose2D:
    stamp: float
    x: float
    y: float
    yaw: float


@dataclass
class LocalPoseSample:
    stamp: float
    x: float
    y: float
    yaw: float


@dataclass
class Correction2D:
    x: float
    y: float
    yaw: float


@dataclass
class Candidate:
    source: str
    msg: Odometry
    pose: Pose2D
    stamp: float
    ok: bool
    quality: float
    reason: str


@dataclass
class LidarStatus:
    source_stamp: float
    received_stamp: float
    ok: bool
    quality: float
    reason: str


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


def pose_from_odom(msg: Odometry) -> Pose2D:
    pose = msg.pose.pose
    return Pose2D(
        stamp=msg.header.stamp.to_sec(),
        x=pose.position.x,
        y=pose.position.y,
        yaw=quaternion_to_yaw(pose.orientation),
    )


def parse_status(status: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in status.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class GlobalLocalizationManager:
    def __init__(self) -> None:
        self.local_odom_topic = rospy.get_param("~local_odom_topic", "/odometry/filtered")
        self.rtk_odom_topic = rospy.get_param("~rtk_odom_topic", "/odometry/rtk_map")
        self.lidar_odom_topic = rospy.get_param("~lidar_odom_topic", "/odometry/lidar_map")
        self.lidar_status_topic = rospy.get_param(
            "~lidar_status_topic", "/lidar_localization/status"
        )
        self.rtk_fix_type_topic = rospy.get_param("~rtk_fix_type_topic", "/rtk/fix_type")
        self.rtk_position_type_topic = rospy.get_param(
            "~rtk_position_type_topic", "/rtk/position_type"
        )
        self.output_topic = rospy.get_param("~output_topic", "/odometry/global_nav")
        self.status_topic = rospy.get_param("~status_topic", "/global_localization/status")

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_link_frame = rospy.get_param("~base_link_frame", "base_link")
        self.publish_tf = bool(rospy.get_param("~publish_tf", True))

        self.min_quality = float(rospy.get_param("~min_quality", 0.5))
        self.source_timeout = float(rospy.get_param("~source_timeout", 0.8))
        self.status_timeout = float(rospy.get_param("~status_timeout", 2.0))
        self.local_interp_timeout = float(rospy.get_param("~local_interp_timeout", 0.2))
        self.switch_cooldown = float(rospy.get_param("~switch_cooldown", 2.0))
        self.switch_hysteresis = float(rospy.get_param("~switch_hysteresis", 0.15))
        self.tie_breaker = str(rospy.get_param("~tie_breaker", "lidar")).strip().lower()
        self.preferred_source = str(rospy.get_param("~preferred_source", "")).strip().lower()
        self.preferred_min_quality = float(rospy.get_param("~preferred_min_quality", 0.95))
        self.fallback_source = str(rospy.get_param("~fallback_source", "")).strip().lower()
        if self.fallback_source == "auto":
            if self.preferred_source == "rtk":
                self.fallback_source = "lidar"
            elif self.preferred_source == "lidar":
                self.fallback_source = "rtk"
            else:
                self.fallback_source = ""
        self.status_stamp_tolerance = float(
            rospy.get_param("~status_stamp_tolerance", 0.02)
        )
        self.max_switch_xy_delta = float(rospy.get_param("~max_switch_xy_delta", 0.5))
        self.max_switch_yaw_delta = math.radians(
            float(rospy.get_param("~max_switch_yaw_delta_deg", 15.0))
        )
        self.smooth_switch_enabled = bool(
            rospy.get_param("~smooth_switch_enabled", True)
        )
        self.switch_blend_duration = float(
            rospy.get_param("~switch_blend_duration", 2.0)
        )
        self.status_interval = float(rospy.get_param("~status_interval", 1.0))
        self.tf_publish_rate = float(rospy.get_param("~tf_publish_rate", 30.0))
        self.transform_time_offset = float(
            rospy.get_param("~transform_time_offset", 0.1)
        )

        self.rtk_good_fix_types = set(rospy.get_param("~rtk_good_fix_types", ["rtk_fixed"]))
        self.rtk_good_position_types = set(
            rospy.get_param("~rtk_good_position_types", ["NARROW_INT", "L1_INT"])
        )
        self.rtk_requires_status = bool(rospy.get_param("~rtk_requires_status", True))

        self.local_history: deque[LocalPoseSample] = deque(maxlen=2000)
        self.latest_rtk: Optional[Candidate] = None
        self.latest_lidar: Optional[Candidate] = None
        self.pending_lidar_msg: Optional[Odometry] = None
        self.lidar_status_history: deque[LidarStatus] = deque(maxlen=100)
        self.rtk_fix_type: Optional[str] = None
        self.rtk_fix_type_stamp: Optional[float] = None
        self.rtk_position_type: Optional[str] = None
        self.rtk_position_type_stamp: Optional[float] = None
        self.lidar_status_ok = False
        self.lidar_quality = 0.0
        self.lidar_status_reason = "waiting_for_lidar_status"
        self.lidar_status_stamp: Optional[float] = None

        self.active_source: Optional[str] = None
        self.last_switch_time = 0.0
        self.last_status_time = 0.0
        self.last_published_source: Optional[str] = None
        self.last_published_stamp: Optional[float] = None
        self.latest_correction_source: Optional[str] = None
        self.latest_correction_source_stamp: Optional[float] = None
        self.latest_correction_x = 0.0
        self.latest_correction_y = 0.0
        self.latest_correction_yaw = 0.0
        self.last_output_pose: Optional[Pose2D] = None
        self.switch_blend_start_time: Optional[float] = None
        self.switch_blend_from_source: Optional[str] = None
        self.switch_blend_to_source: Optional[str] = None
        self.switch_blend_start: Optional[Correction2D] = None
        self.switch_blend_target: Optional[Correction2D] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() if self.publish_tf else None
        self.odom_pub = rospy.Publisher(self.output_topic, Odometry, queue_size=20)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10, latch=True)

        rospy.Subscriber(self.local_odom_topic, Odometry, self.local_odom_callback, queue_size=200)
        rospy.Subscriber(self.rtk_odom_topic, Odometry, self.rtk_odom_callback, queue_size=50)
        rospy.Subscriber(self.lidar_odom_topic, Odometry, self.lidar_odom_callback, queue_size=50)
        rospy.Subscriber(self.lidar_status_topic, String, self.lidar_status_callback, queue_size=20)
        rospy.Subscriber(self.rtk_fix_type_topic, String, self.rtk_fix_type_callback, queue_size=20)
        rospy.Subscriber(
            self.rtk_position_type_topic,
            String,
            self.rtk_position_type_callback,
            queue_size=20,
        )
        rospy.Timer(rospy.Duration(self.status_interval), self.timer_callback)
        if self.tf_broadcaster is not None and self.tf_publish_rate > 0.0:
            rospy.Timer(
                rospy.Duration(1.0 / self.tf_publish_rate),
                self.tf_timer_callback,
            )

        rospy.loginfo(
            "global localization manager started: rtk=%s lidar=%s local=%s output=%s publish_tf=%s tf_rate=%.1f transform_time_offset=%.3f smooth_switch=%s blend_duration=%.2f",
            self.rtk_odom_topic,
            self.lidar_odom_topic,
            self.local_odom_topic,
            self.output_topic,
            self.publish_tf,
            self.tf_publish_rate,
            self.transform_time_offset,
            self.smooth_switch_enabled,
            self.switch_blend_duration,
        )

    def local_odom_callback(self, msg: Odometry) -> None:
        pose = pose_from_odom(msg)
        self.local_history.append(
            LocalPoseSample(stamp=pose.stamp, x=pose.x, y=pose.y, yaw=pose.yaw)
        )

    def rtk_fix_type_callback(self, msg: String) -> None:
        self.rtk_fix_type = msg.data
        self.rtk_fix_type_stamp = rospy.Time.now().to_sec()

    def rtk_position_type_callback(self, msg: String) -> None:
        self.rtk_position_type = msg.data
        self.rtk_position_type_stamp = rospy.Time.now().to_sec()

    def lidar_status_callback(self, msg: String) -> None:
        values = parse_status(msg.data)
        self.lidar_status_ok = values.get("ok", "false").lower() == "true"
        try:
            self.lidar_quality = float(values.get("quality", "0.0"))
        except ValueError:
            self.lidar_quality = 0.0
        self.lidar_status_reason = values.get("reason", "unknown")
        self.lidar_status_stamp = rospy.Time.now().to_sec()

        source_stamp_text = values.get("source_stamp")
        if source_stamp_text is None:
            return

        try:
            source_stamp = float(source_stamp_text)
        except ValueError:
            return

        self.lidar_status_history.append(
            LidarStatus(
                source_stamp=source_stamp,
                received_stamp=self.lidar_status_stamp,
                ok=self.lidar_status_ok,
                quality=self.lidar_quality,
                reason=self.lidar_status_reason,
            )
        )

        if self.pending_lidar_msg is not None:
            pending_stamp = self.pending_lidar_msg.header.stamp.to_sec()
            if abs(pending_stamp - source_stamp) <= self.status_stamp_tolerance:
                pending = self.pending_lidar_msg
                self.pending_lidar_msg = None
                self.process_lidar_odom(pending)

    def interpolate_local_pose(self, stamp: float) -> Optional[LocalPoseSample]:
        if not self.local_history:
            return None

        samples = list(self.local_history)
        if len(samples) == 1:
            only = samples[0]
            if abs(stamp - only.stamp) <= self.local_interp_timeout:
                return only
            return None

        first = samples[0]
        last = samples[-1]
        if stamp <= first.stamp:
            if first.stamp - stamp <= self.local_interp_timeout:
                return first
            return None
        if stamp >= last.stamp:
            if stamp - last.stamp <= self.local_interp_timeout:
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
            if dt > self.local_interp_timeout:
                return None
            ratio = (stamp - previous.stamp) / dt
            return LocalPoseSample(
                stamp=stamp,
                x=previous.x + ratio * (current.x - previous.x),
                y=previous.y + ratio * (current.y - previous.y),
                yaw=wrap_angle(
                    previous.yaw + ratio * wrap_angle(current.yaw - previous.yaw)
                ),
            )

        return None

    def transform_pose_to_map(self, msg: Odometry) -> Optional[Pose2D]:
        source_frame = msg.header.frame_id or self.map_frame
        source_pose = pose_from_odom(msg)
        if source_frame == self.map_frame:
            return source_pose

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Cannot transform %s pose from '%s' to '%s': %s",
                msg.child_frame_id,
                source_frame,
                self.map_frame,
                exc,
            )
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        tf_yaw = quaternion_to_yaw(q)
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        return Pose2D(
            stamp=source_pose.stamp,
            x=t.x + cos_yaw * source_pose.x - sin_yaw * source_pose.y,
            y=t.y + sin_yaw * source_pose.x + cos_yaw * source_pose.y,
            yaw=wrap_angle(tf_yaw + source_pose.yaw),
        )

    def make_map_odom(self, source_msg: Odometry, pose: Pose2D) -> Odometry:
        out = copy.deepcopy(source_msg)
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.base_link_frame
        out.pose.pose.position.x = pose.x
        out.pose.pose.position.y = pose.y
        out.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(pose.yaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        return out

    def rtk_quality(self) -> tuple[bool, float, str]:
        now = rospy.Time.now().to_sec()
        if self.rtk_requires_status:
            if self.rtk_fix_type is None or self.rtk_position_type is None:
                return False, 0.0, "waiting_for_rtk_status"
            if (
                self.rtk_fix_type_stamp is not None
                and now - self.rtk_fix_type_stamp > self.status_timeout
            ):
                return False, 0.0, "stale_rtk_fix_type"
            if (
                self.rtk_position_type_stamp is not None
                and now - self.rtk_position_type_stamp > self.status_timeout
            ):
                return False, 0.0, "stale_rtk_position_type"
            if self.rtk_fix_type not in self.rtk_good_fix_types:
                return False, 0.0, "bad_rtk_fix_type:%s" % self.rtk_fix_type
            if self.rtk_position_type not in self.rtk_good_position_types:
                return False, 0.0, "bad_rtk_position_type:%s" % self.rtk_position_type

        if self.rtk_position_type == "NARROW_INT":
            return True, 1.0, "ok"
        if self.rtk_position_type == "L1_INT":
            return True, 0.8, "ok_l1_int"
        return True, 0.7, "ok_no_status"

    def find_lidar_status(self, source_stamp: float) -> Optional[LidarStatus]:
        best_status = None
        best_delta = None
        for status in self.lidar_status_history:
            delta = abs(status.source_stamp - source_stamp)
            if delta > self.status_stamp_tolerance:
                continue
            if best_delta is None or delta < best_delta:
                best_status = status
                best_delta = delta
        return best_status

    def lidar_quality_state(self, source_stamp: Optional[float] = None) -> tuple[bool, float, str]:
        now = rospy.Time.now().to_sec()
        if source_stamp is not None:
            status = self.find_lidar_status(source_stamp)
            if status is None:
                return False, 0.0, "waiting_for_matching_lidar_status"
            if now - status.received_stamp > self.status_timeout:
                return False, 0.0, "stale_matching_lidar_status"
            if not status.ok:
                return False, status.quality, status.reason
            return True, status.quality, status.reason

        if self.lidar_status_stamp is None:
            return False, 0.0, "waiting_for_lidar_status"
        if now - self.lidar_status_stamp > self.status_timeout:
            return False, 0.0, "stale_lidar_status"
        if not self.lidar_status_ok:
            return False, self.lidar_quality, self.lidar_status_reason
        return True, self.lidar_quality, self.lidar_status_reason

    def rtk_odom_callback(self, msg: Odometry) -> None:
        pose = self.transform_pose_to_map(msg)
        if pose is None:
            return
        ok, quality, reason = self.rtk_quality()
        self.latest_rtk = Candidate("rtk", msg, pose, pose.stamp, ok, quality, reason)
        self.update_from_candidates()

    def lidar_odom_callback(self, msg: Odometry) -> None:
        if self.find_lidar_status(msg.header.stamp.to_sec()) is None:
            self.pending_lidar_msg = msg
            self.publish_status("waiting_for_matching_lidar_status")
            return
        self.process_lidar_odom(msg)

    def process_lidar_odom(self, msg: Odometry) -> None:
        pose = self.transform_pose_to_map(msg)
        if pose is None:
            return
        ok, quality, reason = self.lidar_quality_state(pose.stamp)
        self.latest_lidar = Candidate("lidar", msg, pose, pose.stamp, ok, quality, reason)
        self.update_from_candidates()

    def candidate_is_fresh(self, candidate: Candidate, now: float) -> bool:
        return now - candidate.stamp <= self.source_timeout

    def eligible_candidates(self) -> list[Candidate]:
        now = rospy.Time.now().to_sec()
        items = []
        for candidate in (self.latest_lidar, self.latest_rtk):
            if candidate is None:
                continue
            if not self.candidate_is_fresh(candidate, now):
                continue
            if not candidate.ok or candidate.quality < self.min_quality:
                continue
            items.append(candidate)
        return items

    def eligible_candidate_for_source(self, source: str) -> Optional[Candidate]:
        for candidate in self.eligible_candidates():
            if candidate.source == source:
                return candidate
        return None

    def choose_candidate(self) -> Optional[Candidate]:
        candidates = self.eligible_candidates()
        if not candidates:
            return None

        by_source = {candidate.source: candidate for candidate in candidates}
        current = by_source.get(self.active_source or "")
        best = max(candidates, key=lambda item: item.quality)
        now = rospy.Time.now().to_sec()

        preferred = by_source.get(self.preferred_source)
        if preferred is not None and preferred.quality >= self.preferred_min_quality:
            if current is None or preferred.source == current.source:
                return preferred
            if now - self.last_switch_time >= self.switch_cooldown:
                return preferred

        fallback = by_source.get(self.fallback_source)
        if (
            (preferred is None or preferred.quality < self.preferred_min_quality)
            and fallback is not None
        ):
            if current is None or fallback.source == current.source:
                return fallback
            if now - self.last_switch_time >= self.switch_cooldown:
                return fallback

        if current is not None:
            if now - self.last_switch_time < self.switch_cooldown:
                return current
            if best.source != current.source and best.quality < current.quality + self.switch_hysteresis:
                return current

        if len(candidates) > 1:
            top_quality = best.quality
            tied = [
                item
                for item in candidates
                if abs(item.quality - top_quality) <= self.switch_hysteresis
            ]
            if current is not None and current in tied:
                return current
            if self.tie_breaker in by_source and by_source[self.tie_breaker] in tied:
                return by_source[self.tie_breaker]

        return best

    def switch_is_safe(self, candidate: Candidate) -> tuple[bool, str]:
        if self.active_source is None or candidate.source == self.active_source:
            return True, "ok"
        if self.last_output_pose is None:
            return True, "ok"

        xy_delta = math.hypot(
            candidate.pose.x - self.last_output_pose.x,
            candidate.pose.y - self.last_output_pose.y,
        )
        yaw_delta = abs(wrap_angle(candidate.pose.yaw - self.last_output_pose.yaw))

        if self.max_switch_xy_delta > 0.0 and xy_delta > self.max_switch_xy_delta:
            return (
                False,
                "switch_rejected_large_xy_delta:%.3f" % xy_delta,
            )
        if self.max_switch_yaw_delta > 0.0 and yaw_delta > self.max_switch_yaw_delta:
            return (
                False,
                "switch_rejected_large_yaw_delta_deg:%.2f" % math.degrees(yaw_delta),
            )

        return True, "ok"

    def correction_from_pose(
        self, pose: Pose2D, local_pose: LocalPoseSample
    ) -> Correction2D:
        yaw_correction = wrap_angle(pose.yaw - local_pose.yaw)
        cos_yaw = math.cos(yaw_correction)
        sin_yaw = math.sin(yaw_correction)
        return Correction2D(
            x=pose.x - (cos_yaw * local_pose.x - sin_yaw * local_pose.y),
            y=pose.y - (sin_yaw * local_pose.x + cos_yaw * local_pose.y),
            yaw=yaw_correction,
        )

    def pose_from_correction(
        self, correction: Correction2D, local_pose: LocalPoseSample
    ) -> Pose2D:
        cos_yaw = math.cos(correction.yaw)
        sin_yaw = math.sin(correction.yaw)
        return Pose2D(
            stamp=local_pose.stamp,
            x=correction.x + cos_yaw * local_pose.x - sin_yaw * local_pose.y,
            y=correction.y + sin_yaw * local_pose.x + cos_yaw * local_pose.y,
            yaw=wrap_angle(correction.yaw + local_pose.yaw),
        )

    def current_correction(self) -> Correction2D:
        return Correction2D(
            x=self.latest_correction_x,
            y=self.latest_correction_y,
            yaw=self.latest_correction_yaw,
        )

    def clear_smooth_switch(self) -> None:
        self.switch_blend_start_time = None
        self.switch_blend_from_source = None
        self.switch_blend_to_source = None
        self.switch_blend_start = None
        self.switch_blend_target = None

    def smooth_switch_is_active(self, now: Optional[float] = None) -> bool:
        if (
            self.switch_blend_start_time is None
            or self.switch_blend_start is None
            or self.switch_blend_target is None
        ):
            return False
        if self.switch_blend_duration <= 0.0:
            return False
        if now is None:
            now = rospy.Time.now().to_sec()
        return now - self.switch_blend_start_time < self.switch_blend_duration

    def start_smooth_switch(
        self,
        from_source: Optional[str],
        to_source: str,
        target_correction: Correction2D,
    ) -> bool:
        if (
            not self.smooth_switch_enabled
            or self.switch_blend_duration <= 0.0
            or from_source is None
            or self.latest_correction_source_stamp is None
        ):
            self.clear_smooth_switch()
            return False

        self.switch_blend_start_time = rospy.Time.now().to_sec()
        self.switch_blend_from_source = from_source
        self.switch_blend_to_source = to_source
        self.switch_blend_start = self.current_correction()
        self.switch_blend_target = target_correction
        return True

    def update_smooth_correction(self, now: Optional[float] = None) -> bool:
        if (
            self.switch_blend_start_time is None
            or self.switch_blend_start is None
            or self.switch_blend_target is None
        ):
            return False
        if self.switch_blend_duration <= 0.0:
            self.clear_smooth_switch()
            return False
        if now is None:
            now = rospy.Time.now().to_sec()

        ratio = (now - self.switch_blend_start_time) / self.switch_blend_duration
        if ratio >= 1.0:
            self.latest_correction_x = self.switch_blend_target.x
            self.latest_correction_y = self.switch_blend_target.y
            self.latest_correction_yaw = self.switch_blend_target.yaw
            self.clear_smooth_switch()
            return False
        if ratio <= 0.0:
            ratio = 0.0

        eased = ratio * ratio * (3.0 - 2.0 * ratio)
        start = self.switch_blend_start
        target = self.switch_blend_target
        self.latest_correction_x = start.x + eased * (target.x - start.x)
        self.latest_correction_y = start.y + eased * (target.y - start.y)
        self.latest_correction_yaw = wrap_angle(
            start.yaw + eased * wrap_angle(target.yaw - start.yaw)
        )
        return True

    def smooth_switch_summary(self, now: Optional[float] = None) -> str:
        if not self.smooth_switch_is_active(now):
            return "none"
        if now is None:
            now = rospy.Time.now().to_sec()
        progress = (now - (self.switch_blend_start_time or now)) / self.switch_blend_duration
        progress = max(0.0, min(1.0, progress))
        return "%s->%s,progress=%.2f" % (
            self.switch_blend_from_source or "none",
            self.switch_blend_to_source or "none",
            progress,
        )

    def publish_selected(
        self,
        candidate: Candidate,
        local_pose: LocalPoseSample,
        target_correction: Correction2D,
    ) -> Pose2D:
        yaw_correction = wrap_angle(candidate.pose.yaw - local_pose.yaw)
        now = rospy.Time.now().to_sec()

        if (
            self.smooth_switch_is_active(now)
            and self.switch_blend_to_source == candidate.source
        ):
            self.switch_blend_target = target_correction
            self.update_smooth_correction(now)
            published_correction = self.current_correction()
        else:
            self.clear_smooth_switch()
            published_correction = target_correction
            self.latest_correction_x = target_correction.x
            self.latest_correction_y = target_correction.y
            self.latest_correction_yaw = yaw_correction

        output_pose = self.pose_from_correction(published_correction, local_pose)
        output_msg = self.make_map_odom(candidate.msg, output_pose)
        self.odom_pub.publish(output_msg)

        self.latest_correction_source = candidate.source
        self.latest_correction_source_stamp = candidate.stamp
        self.publish_cached_tf()
        return output_pose

    def publish_cached_tf(self) -> None:
        if self.tf_broadcaster is None or self.latest_correction_source_stamp is None:
            return

        now = rospy.Time.now()
        self.update_smooth_correction(now.to_sec())
        if now.to_sec() - self.latest_correction_source_stamp > self.source_timeout:
            return

        transform = TransformStamped()
        transform.header.stamp = now + rospy.Duration(self.transform_time_offset)
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.odom_frame
        transform.transform.translation.x = self.latest_correction_x
        transform.transform.translation.y = self.latest_correction_y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = math.sin(0.5 * self.latest_correction_yaw)
        transform.transform.rotation.w = math.cos(0.5 * self.latest_correction_yaw)
        self.tf_broadcaster.sendTransform(transform)

    def update_from_candidates(self) -> None:
        candidate = self.choose_candidate()
        if candidate is None:
            self.publish_status("holding_no_valid_source")
            return

        if self.active_source != candidate.source:
            safe, switch_reason = self.switch_is_safe(candidate)
            if not safe:
                self.publish_status(switch_reason)
                current = self.eligible_candidate_for_source(self.active_source or "")
                if current is None:
                    return
                candidate = current

        local_pose = self.interpolate_local_pose(candidate.stamp)
        if local_pose is None:
            self.publish_status("holding_no_local_pose_at_source_stamp")
            return

        target_correction = self.correction_from_pose(candidate.pose, local_pose)
        if self.active_source != candidate.source:
            previous_source = self.active_source
            smooth_started = self.start_smooth_switch(
                previous_source, candidate.source, target_correction
            )
            rospy.loginfo(
                "Switching global localization source: %s -> %s (quality=%.3f reason=%s smooth=%s)",
                previous_source or "none",
                candidate.source,
                candidate.quality,
                candidate.reason,
                smooth_started,
            )
            self.active_source = candidate.source
            self.last_switch_time = rospy.Time.now().to_sec()

        if (
            self.last_published_source == candidate.source
            and self.last_published_stamp is not None
            and abs(self.last_published_stamp - candidate.stamp) < 1e-9
            and not self.smooth_switch_is_active()
        ):
            self.publish_status("ok")
            return

        output_pose = self.publish_selected(candidate, local_pose, target_correction)
        self.last_published_source = candidate.source
        self.last_published_stamp = candidate.stamp
        self.last_output_pose = output_pose
        self.publish_status("ok")

    def candidate_summary(self, candidate: Optional[Candidate], now: float) -> str:
        if candidate is None:
            return "none"
        age = now - candidate.stamp
        return "%s:ok=%s,q=%.3f,age=%.2f,reason=%s" % (
            candidate.source,
            str(candidate.ok).lower(),
            candidate.quality,
            age,
            candidate.reason,
        )

    def publish_status(self, reason: str) -> None:
        now = rospy.Time.now().to_sec()
        if now - self.last_status_time < self.status_interval:
            return
        self.last_status_time = now
        status = "active=%s reason=%s smooth=%s %s %s" % (
            self.active_source or "none",
            reason,
            self.smooth_switch_summary(now),
            self.candidate_summary(self.latest_lidar, now),
            self.candidate_summary(self.latest_rtk, now),
        )
        self.status_pub.publish(String(data=status))

    def timer_callback(self, _event) -> None:
        self.update_from_candidates()

    def tf_timer_callback(self, _event) -> None:
        self.publish_cached_tf()


if __name__ == "__main__":
    rospy.init_node("global_localization_manager")
    GlobalLocalizationManager()
    rospy.spin()

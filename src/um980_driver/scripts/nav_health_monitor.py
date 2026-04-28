#!/usr/bin/env python3
"""Monitor navigation health and record short, useful diagnostics.

The node is intentionally passive: it only subscribes to existing navigation
topics, publishes status/markers, and writes CSV logs. It never sends goals or
velocity commands.
"""

from __future__ import annotations

import csv
import math
import os
import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import rospy
import tf2_ros
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Point, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


STATUS_NAMES = {
    GoalStatus.PENDING: "PENDING",
    GoalStatus.ACTIVE: "ACTIVE",
    GoalStatus.PREEMPTED: "PREEMPTED",
    GoalStatus.SUCCEEDED: "SUCCEEDED",
    GoalStatus.ABORTED: "ABORTED",
    GoalStatus.REJECTED: "REJECTED",
    GoalStatus.PREEMPTING: "PREEMPTING",
    GoalStatus.RECALLING: "RECALLING",
    GoalStatus.RECALLED: "RECALLED",
    GoalStatus.LOST: "LOST",
}


@dataclass
class Pose2D:
    stamp: float
    received: float
    frame_id: str
    x: float
    y: float
    yaw: float


@dataclass
class Twist2D:
    received: float
    linear_x: float
    angular_z: float


@dataclass
class StatusSample:
    received: float
    text: str
    fields: Dict[str, str]


@dataclass
class PlanSample:
    received: float
    frame_id: str
    size: int
    length: float


@dataclass
class EventSample:
    stamp: float
    level: str
    event: str
    detail: str


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


def pose_from_odom(msg: Odometry, received: float) -> Pose2D:
    pose = msg.pose.pose
    return Pose2D(
        stamp=msg.header.stamp.to_sec() if msg.header.stamp else received,
        received=received,
        frame_id=msg.header.frame_id,
        x=pose.position.x,
        y=pose.position.y,
        yaw=quaternion_to_yaw(pose.orientation),
    )


def parse_status_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for part in text.split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def parse_patrol_index(text: str, field: str) -> Optional[int]:
    match = re.search(r"%s=index=(\d+)" % re.escape(field), text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "unknown":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def path_length(path: Path) -> float:
    length = 0.0
    previous = None
    for pose in path.poses:
        point = pose.pose.position
        if previous is not None:
            length += math.hypot(point.x - previous.x, point.y - previous.y)
        previous = point
    return length


class NavHealthMonitor:
    def __init__(self) -> None:
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.global_odom_topic = rospy.get_param("~global_odom_topic", "/odometry/global_nav")
        self.rtk_odom_topic = rospy.get_param("~rtk_odom_topic", "/odometry/rtk_map")
        self.lidar_odom_topic = rospy.get_param("~lidar_odom_topic", "/odometry/lidar_map")
        self.local_odom_topic = rospy.get_param("~local_odom_topic", "/odometry/filtered")
        self.chassis_odom_topic = rospy.get_param(
            "~chassis_odom_topic", "/ranger_base_node/odom"
        )
        self.cmd_vel_nav_topic = rospy.get_param("~cmd_vel_nav_topic", "/cmd_vel_nav")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        self.global_status_topic = rospy.get_param(
            "~global_status_topic", "/global_localization/status"
        )
        self.lidar_status_topic = rospy.get_param(
            "~lidar_status_topic", "/lidar_localization/status"
        )
        self.relocalizer_status_topic = rospy.get_param(
            "~relocalizer_status_topic", "/rtk_lidar_relocalizer/status"
        )
        self.waypoint_status_topic = rospy.get_param(
            "~waypoint_status_topic", "/waypoint_patrol/status"
        )
        self.rtk_fix_type_topic = rospy.get_param("~rtk_fix_type_topic", "/rtk/fix_type")
        self.rtk_position_type_topic = rospy.get_param(
            "~rtk_position_type_topic", "/rtk/position_type"
        )
        self.move_base_status_topic = rospy.get_param(
            "~move_base_status_topic", "/move_base/status"
        )
        self.global_plan_topic = rospy.get_param(
            "~global_plan_topic", "/move_base/NavfnROS/plan"
        )
        self.local_plan_topic = rospy.get_param(
            "~local_plan_topic", "/move_base/DWAPlannerROS/local_plan"
        )
        self.rpp_status_topic = rospy.get_param(
            "~rpp_status_topic", "/move_base/RegulatedPurePursuitPlanner/status"
        )
        self.rpp_local_plan_topic = rospy.get_param(
            "~rpp_local_plan_topic", "/move_base/RegulatedPurePursuitPlanner/local_plan"
        )

        self.status_topic = rospy.get_param("~status_topic", "/nav_health/status")
        self.event_topic = rospy.get_param("~event_topic", "/nav_health/events")
        self.marker_topic = rospy.get_param("~marker_topic", "/nav_health/markers")
        self.marker_frame = rospy.get_param("~marker_frame", self.base_frame)

        self.status_rate = float(rospy.get_param("~status_rate", 2.0))
        self.snapshot_rate = float(rospy.get_param("~snapshot_rate", 1.0))
        self.topic_timeout = float(rospy.get_param("~topic_timeout", 2.0))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.05))
        self.event_hold_time = float(rospy.get_param("~event_hold_time", 5.0))
        self.event_cooldown = float(rospy.get_param("~event_cooldown", 3.0))

        self.rtk_lidar_xy_warn = float(rospy.get_param("~rtk_lidar_xy_warn", 0.8))
        self.rtk_lidar_yaw_warn = math.radians(
            float(rospy.get_param("~rtk_lidar_yaw_warn_deg", 20.0))
        )
        self.pose_jump_xy = float(rospy.get_param("~pose_jump_xy", 0.35))
        self.pose_jump_speed = float(rospy.get_param("~pose_jump_speed", 1.5))
        self.pose_jump_yaw = math.radians(float(rospy.get_param("~pose_jump_yaw_deg", 20.0)))
        self.pose_jump_yaw_rate = math.radians(
            float(rospy.get_param("~pose_jump_yaw_rate_deg", 120.0))
        )
        self.cmd_oscillation_window = float(rospy.get_param("~cmd_oscillation_window", 2.0))
        self.cmd_oscillation_min_sign_changes = int(
            rospy.get_param("~cmd_oscillation_min_sign_changes", 4)
        )
        self.cmd_oscillation_min_abs_w = float(
            rospy.get_param("~cmd_oscillation_min_abs_w", 0.12)
        )
        self.commanded_motion_v = float(rospy.get_param("~commanded_motion_v", 0.10))
        self.commanded_motion_w = float(rospy.get_param("~commanded_motion_w", 0.15))
        self.actual_motion_v = float(rospy.get_param("~actual_motion_v", 0.03))
        self.actual_motion_w = float(rospy.get_param("~actual_motion_w", 0.05))
        self.command_mismatch_duration = float(
            rospy.get_param("~command_mismatch_duration", 1.0)
        )

        self.log_enabled = bool(rospy.get_param("~log_enabled", True))
        default_log_dir = os.path.join(os.path.expanduser("~"), ".ros", "nav_health")
        self.log_dir = os.path.expanduser(rospy.get_param("~log_dir", default_log_dir))

        self.lock = threading.Lock()
        self.global_pose: Optional[Pose2D] = None
        self.previous_global_pose: Optional[Pose2D] = None
        self.rtk_pose: Optional[Pose2D] = None
        self.lidar_pose: Optional[Pose2D] = None
        self.local_pose: Optional[Pose2D] = None
        self.chassis_pose: Optional[Pose2D] = None
        self.chassis_twist: Optional[Twist2D] = None
        self.cmd_vel_nav: Optional[Twist2D] = None
        self.cmd_vel: Optional[Twist2D] = None

        self.global_status: Optional[StatusSample] = None
        self.lidar_status: Optional[StatusSample] = None
        self.relocalizer_status: Optional[StatusSample] = None
        self.waypoint_status: Optional[StatusSample] = None
        self.rpp_status: Optional[StatusSample] = None
        self.rtk_fix_type: Optional[StatusSample] = None
        self.rtk_position_type: Optional[StatusSample] = None

        self.move_base_status = "none"
        self.move_base_text = ""
        self.move_base_received = 0.0
        self.last_move_base_terminal_key: Optional[Tuple[str, int]] = None
        self.global_plan: Optional[PlanSample] = None
        self.local_plan: Optional[PlanSample] = None

        self.active_source = "none"
        self.previous_active_source = "none"
        self.waypoint_index: Optional[int] = None
        self.previous_waypoint_index: Optional[int] = None
        self.cmd_nav_history: Deque[Tuple[float, float]] = deque(maxlen=200)
        self.cmd_history: Deque[Tuple[float, float]] = deque(maxlen=200)
        self.recent_events: Deque[EventSample] = deque(maxlen=100)
        self.last_event_by_type: Dict[str, float] = {}
        self.command_mismatch_since: Optional[float] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10, latch=True)
        self.event_pub = rospy.Publisher(self.event_topic, String, queue_size=20)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        self.snapshot_writer: Optional[csv.DictWriter] = None
        self.event_writer: Optional[csv.DictWriter] = None
        self.snapshot_file = None
        self.event_file = None
        self.setup_logs()

        self.setup_subscribers()
        self.status_timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.status_rate, 0.1)), self.status_timer_callback
        )
        self.snapshot_timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.snapshot_rate, 0.1)), self.snapshot_timer_callback
        )

        rospy.on_shutdown(self.close_logs)
        rospy.loginfo(
            "nav health monitor started: status=%s events=%s markers=%s logs=%s",
            self.status_topic,
            self.event_topic,
            self.marker_topic,
            self.log_dir if self.log_enabled else "disabled",
        )

    def setup_subscribers(self) -> None:
        rospy.Subscriber(self.global_odom_topic, Odometry, self.global_odom_callback, queue_size=20)
        rospy.Subscriber(self.rtk_odom_topic, Odometry, self.rtk_odom_callback, queue_size=20)
        rospy.Subscriber(self.lidar_odom_topic, Odometry, self.lidar_odom_callback, queue_size=20)
        rospy.Subscriber(self.local_odom_topic, Odometry, self.local_odom_callback, queue_size=20)
        rospy.Subscriber(self.chassis_odom_topic, Odometry, self.chassis_odom_callback, queue_size=20)
        rospy.Subscriber(self.cmd_vel_nav_topic, Twist, self.cmd_vel_nav_callback, queue_size=50)
        rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_callback, queue_size=50)
        rospy.Subscriber(self.global_status_topic, String, self.global_status_callback, queue_size=20)
        rospy.Subscriber(self.lidar_status_topic, String, self.lidar_status_callback, queue_size=20)
        rospy.Subscriber(
            self.relocalizer_status_topic, String, self.relocalizer_status_callback, queue_size=20
        )
        rospy.Subscriber(
            self.waypoint_status_topic, String, self.waypoint_status_callback, queue_size=20
        )
        rospy.Subscriber(self.rtk_fix_type_topic, String, self.rtk_fix_type_callback, queue_size=20)
        rospy.Subscriber(
            self.rtk_position_type_topic, String, self.rtk_position_type_callback, queue_size=20
        )
        rospy.Subscriber(
            self.move_base_status_topic, GoalStatusArray, self.move_base_status_callback, queue_size=20
        )
        rospy.Subscriber(self.global_plan_topic, Path, self.global_plan_callback, queue_size=5)
        rospy.Subscriber(self.local_plan_topic, Path, self.local_plan_callback, queue_size=5)
        rospy.Subscriber(self.rpp_status_topic, String, self.rpp_status_callback, queue_size=20)
        rospy.Subscriber(
            self.rpp_local_plan_topic, Path, self.local_plan_callback, queue_size=5
        )

    def setup_logs(self) -> None:
        if not self.log_enabled:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = rospy.Time.now().to_sec()
        suffix = str(int(timestamp)) if timestamp > 0.0 else "startup"
        snapshot_path = os.path.join(self.log_dir, "nav_health_%s.csv" % suffix)
        event_path = os.path.join(self.log_dir, "nav_health_events_%s.csv" % suffix)

        snapshot_fields = [
            "time",
            "level",
            "last_event",
            "active_source",
            "rtk_fix_type",
            "rtk_position_type",
            "rtk_lidar_xy",
            "rtk_lidar_yaw_deg",
            "global_pose_age",
            "rtk_pose_age",
            "lidar_pose_age",
            "cmd_nav_v",
            "cmd_nav_w",
            "cmd_v",
            "cmd_w",
            "odom_v",
            "odom_w",
            "waypoint_state",
            "waypoint_index",
            "route_error",
            "checkpoint_distance",
            "move_base_status",
            "local_controller",
            "rpp_reason",
            "rpp_status_age",
            "rpp_lookahead",
            "rpp_target_x",
            "rpp_target_y",
            "rpp_curvature",
            "rpp_linear",
            "rpp_angular",
            "rpp_remaining",
            "global_plan_size",
            "global_plan_length",
            "local_plan_size",
            "local_plan_length",
            "map_odom_tf_age",
            "odom_base_tf_age",
            "map_base_tf_age",
            "summary",
        ]
        event_fields = ["time", "level", "event", "detail"]

        self.snapshot_file = open(snapshot_path, "w", newline="", encoding="utf-8")
        self.event_file = open(event_path, "w", newline="", encoding="utf-8")
        self.snapshot_writer = csv.DictWriter(self.snapshot_file, fieldnames=snapshot_fields)
        self.event_writer = csv.DictWriter(self.event_file, fieldnames=event_fields)
        self.snapshot_writer.writeheader()
        self.event_writer.writeheader()

    def close_logs(self) -> None:
        for handle in (self.snapshot_file, self.event_file):
            if handle is not None:
                handle.flush()
                handle.close()

    def global_odom_callback(self, msg: Odometry) -> None:
        now = rospy.Time.now().to_sec()
        pose = pose_from_odom(msg, now)
        with self.lock:
            previous = self.global_pose
            self.previous_global_pose = previous
            self.global_pose = pose
        self.detect_pose_jump(previous, pose)

    def rtk_odom_callback(self, msg: Odometry) -> None:
        with self.lock:
            self.rtk_pose = pose_from_odom(msg, rospy.Time.now().to_sec())

    def lidar_odom_callback(self, msg: Odometry) -> None:
        with self.lock:
            self.lidar_pose = pose_from_odom(msg, rospy.Time.now().to_sec())

    def local_odom_callback(self, msg: Odometry) -> None:
        with self.lock:
            self.local_pose = pose_from_odom(msg, rospy.Time.now().to_sec())

    def chassis_odom_callback(self, msg: Odometry) -> None:
        now = rospy.Time.now().to_sec()
        with self.lock:
            self.chassis_pose = pose_from_odom(msg, now)
            self.chassis_twist = Twist2D(
                received=now,
                linear_x=msg.twist.twist.linear.x,
                angular_z=msg.twist.twist.angular.z,
            )

    def cmd_vel_nav_callback(self, msg: Twist) -> None:
        self.update_cmd_sample(msg, is_nav=True)

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.update_cmd_sample(msg, is_nav=False)

    def update_cmd_sample(self, msg: Twist, is_nav: bool) -> None:
        now = rospy.Time.now().to_sec()
        sample = Twist2D(received=now, linear_x=msg.linear.x, angular_z=msg.angular.z)
        history = self.cmd_nav_history if is_nav else self.cmd_history
        with self.lock:
            if is_nav:
                self.cmd_vel_nav = sample
            else:
                self.cmd_vel = sample
            history.append((now, msg.angular.z))
        self.detect_cmd_oscillation(history, "cmd_vel_nav" if is_nav else "cmd_vel")

    def make_status_sample(self, msg: String) -> StatusSample:
        now = rospy.Time.now().to_sec()
        return StatusSample(received=now, text=msg.data, fields=parse_status_fields(msg.data))

    def global_status_callback(self, msg: String) -> None:
        sample = self.make_status_sample(msg)
        active = sample.fields.get("active", "none")
        with self.lock:
            previous = self.active_source
            self.global_status = sample
            self.previous_active_source = previous
            self.active_source = active
        if active != "none" and previous != "none" and active != previous:
            self.add_event("WARN", "source_switch", "%s -> %s" % (previous, active))

    def lidar_status_callback(self, msg: String) -> None:
        with self.lock:
            self.lidar_status = self.make_status_sample(msg)

    def relocalizer_status_callback(self, msg: String) -> None:
        sample = self.make_status_sample(msg)
        with self.lock:
            self.relocalizer_status = sample
        reason = sample.fields.get("reason", "")
        if "published_initialpose" in reason or reason.startswith("auto_"):
            self.add_event("WARN", "relocalize", sample.text)

    def waypoint_status_callback(self, msg: String) -> None:
        sample = self.make_status_sample(msg)
        current_index = parse_patrol_index(msg.data, "checkpoint")
        with self.lock:
            previous = self.waypoint_index
            self.waypoint_status = sample
            self.previous_waypoint_index = previous
            self.waypoint_index = current_index
        if current_index is not None and previous is not None and current_index != previous:
            self.add_event("INFO", "waypoint_changed", "%s -> %s" % (previous, current_index))

    def rpp_status_callback(self, msg: String) -> None:
        sample = self.make_status_sample(msg)
        reason = sample.fields.get("reason", "")
        detail = msg.data
        with self.lock:
            self.rpp_status = sample
        if reason == "obstacle":
            self.add_event("WARN", "rpp_obstacle", detail)
        elif reason == "target_behind":
            self.add_event("WARN", "rpp_target_behind", detail)

    def rtk_fix_type_callback(self, msg: String) -> None:
        with self.lock:
            self.rtk_fix_type = self.make_status_sample(msg)

    def rtk_position_type_callback(self, msg: String) -> None:
        with self.lock:
            self.rtk_position_type = self.make_status_sample(msg)

    def move_base_status_callback(self, msg: GoalStatusArray) -> None:
        now = rospy.Time.now().to_sec()
        chosen = None
        for status in msg.status_list:
            if status.status == GoalStatus.ACTIVE:
                chosen = status
                break
        if chosen is None and msg.status_list:
            chosen = msg.status_list[-1]

        with self.lock:
            self.move_base_received = now
            if chosen is None:
                self.move_base_status = "none"
                self.move_base_text = ""
            else:
                self.move_base_status = STATUS_NAMES.get(chosen.status, str(chosen.status))
                self.move_base_text = chosen.text

        if chosen is not None and chosen.status in (
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.LOST,
        ):
            terminal_key = (chosen.goal_id.id, chosen.status)
            if terminal_key == self.last_move_base_terminal_key:
                return
            self.last_move_base_terminal_key = terminal_key
            self.add_event(
                "ERROR",
                "move_base_%s" % STATUS_NAMES.get(chosen.status, chosen.status).lower(),
                chosen.text,
            )

    def global_plan_callback(self, msg: Path) -> None:
        with self.lock:
            self.global_plan = PlanSample(
                received=rospy.Time.now().to_sec(),
                frame_id=msg.header.frame_id,
                size=len(msg.poses),
                length=path_length(msg),
            )

    def local_plan_callback(self, msg: Path) -> None:
        with self.lock:
            self.local_plan = PlanSample(
                received=rospy.Time.now().to_sec(),
                frame_id=msg.header.frame_id,
                size=len(msg.poses),
                length=path_length(msg),
            )

    def add_event(self, level: str, event: str, detail: str) -> None:
        now = rospy.Time.now().to_sec()
        cooldown_key = event
        if event in ("source_switch", "waypoint_changed", "relocalize"):
            cooldown_key = "%s:%s" % (event, detail[:80])
        last = self.last_event_by_type.get(cooldown_key, 0.0)
        if now - last < self.event_cooldown:
            return
        self.last_event_by_type[cooldown_key] = now
        detail = detail.replace("\n", " ")[:500]
        sample = EventSample(stamp=now, level=level, event=event, detail=detail)
        with self.lock:
            self.recent_events.append(sample)
        self.event_pub.publish(String(data=self.format_event(sample)))
        if self.event_writer is not None and self.event_file is not None:
            self.event_writer.writerow(
                {"time": now, "level": level, "event": event, "detail": detail}
            )
            self.event_file.flush()

    def detect_pose_jump(self, previous: Optional[Pose2D], current: Pose2D) -> None:
        if previous is None:
            return
        dt = current.received - previous.received
        if dt <= 0.02 or dt > 1.0:
            return
        xy_step = math.hypot(current.x - previous.x, current.y - previous.y)
        yaw_step = abs(wrap_angle(current.yaw - previous.yaw))
        speed = xy_step / dt
        yaw_rate = yaw_step / dt
        xy_jump = xy_step > self.pose_jump_xy and speed > self.pose_jump_speed
        yaw_jump = yaw_step > self.pose_jump_yaw and yaw_rate > self.pose_jump_yaw_rate
        if xy_jump or yaw_jump:
            with self.lock:
                source = self.active_source
                fix = self.rtk_fix_type.text if self.rtk_fix_type is not None else "unknown"
            self.add_event(
                "WARN",
                "global_pose_jump",
                "xy_step=%.3f speed=%.2f yaw_step_deg=%.1f yaw_rate_deg=%.1f dt=%.3f source=%s rtk=%s"
                % (xy_step, speed, math.degrees(yaw_step), math.degrees(yaw_rate), dt, source, fix),
            )

    def detect_cmd_oscillation(
        self, history: Deque[Tuple[float, float]], event_prefix: str
    ) -> None:
        now = rospy.Time.now().to_sec()
        values = [
            w
            for stamp, w in history
            if now - stamp <= self.cmd_oscillation_window
            and abs(w) >= self.cmd_oscillation_min_abs_w
        ]
        if len(values) < self.cmd_oscillation_min_sign_changes + 1:
            return

        sign_changes = 0
        last_sign = 0
        for value in values:
            sign = 1 if value > 0.0 else -1
            if last_sign != 0 and sign != last_sign:
                sign_changes += 1
            last_sign = sign

        if sign_changes >= self.cmd_oscillation_min_sign_changes:
            self.add_event(
                "WARN",
                "%s_oscillation" % event_prefix,
                "sign_changes=%d window=%.1fs latest_w=%.3f"
                % (sign_changes, self.cmd_oscillation_window, values[-1]),
            )

    def lookup_tf_age(self, target: str, source: str, now: float) -> Tuple[str, Optional[float]]:
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target, source, rospy.Time(0), rospy.Duration(self.tf_timeout)
            )
        except Exception as exc:  # pylint: disable=broad-except
            return "tf_error:%s->%s:%s" % (target, source, str(exc)), None

        stamp = transform.header.stamp.to_sec()
        if stamp <= 0.0:
            return "ok", None
        return "ok", now - stamp

    def topic_age(self, sample) -> Optional[float]:
        if sample is None:
            return None
        return rospy.Time.now().to_sec() - sample.received

    def transform_pose_to_frame(
        self, pose: Pose2D, target_frame: str
    ) -> Optional[Pose2D]:
        source_frame = pose.frame_id or target_frame
        if source_frame == target_frame:
            return pose

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout),
            )
        except Exception as exc:  # pylint: disable=broad-except
            rospy.logwarn_throttle(
                5.0,
                "Cannot transform nav health pose from '%s' to '%s': %s",
                source_frame,
                target_frame,
                exc,
            )
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        tf_yaw = quaternion_to_yaw(q)
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        return Pose2D(
            stamp=pose.stamp,
            received=pose.received,
            frame_id=target_frame,
            x=t.x + cos_yaw * pose.x - sin_yaw * pose.y,
            y=t.y + sin_yaw * pose.x + cos_yaw * pose.y,
            yaw=wrap_angle(tf_yaw + pose.yaw),
        )

    def compute_rtk_lidar_delta(self) -> Tuple[Optional[float], Optional[float]]:
        with self.lock:
            raw_rtk = self.rtk_pose
            raw_lidar = self.lidar_pose
        if raw_rtk is None or raw_lidar is None:
            return None, None

        rtk = self.transform_pose_to_frame(raw_rtk, self.map_frame)
        lidar = self.transform_pose_to_frame(raw_lidar, self.map_frame)
        if rtk is None or lidar is None:
            return None, None

        xy = math.hypot(rtk.x - lidar.x, rtk.y - lidar.y)
        yaw = abs(wrap_angle(rtk.yaw - lidar.yaw))
        return xy, yaw

    def check_periodic_events(self, now: float) -> None:
        xy, yaw = self.compute_rtk_lidar_delta()
        if xy is not None and yaw is not None:
            if xy > self.rtk_lidar_xy_warn or yaw > self.rtk_lidar_yaw_warn:
                self.add_event(
                    "WARN",
                    "rtk_lidar_delta",
                    "xy=%.3f yaw_deg=%.1f" % (xy, math.degrees(yaw)),
                )

        with self.lock:
            cmd = self.cmd_vel
            odom = self.chassis_twist
        if cmd is None or odom is None:
            self.command_mismatch_since = None
            return
        cmd_fresh = now - cmd.received <= self.topic_timeout
        odom_fresh = now - odom.received <= self.topic_timeout
        commanded = (
            abs(cmd.linear_x) >= self.commanded_motion_v
            or abs(cmd.angular_z) >= self.commanded_motion_w
        )
        actual_stopped = (
            abs(odom.linear_x) <= self.actual_motion_v
            and abs(odom.angular_z) <= self.actual_motion_w
        )
        if cmd_fresh and odom_fresh and commanded and actual_stopped:
            if self.command_mismatch_since is None:
                self.command_mismatch_since = now
            elif now - self.command_mismatch_since >= self.command_mismatch_duration:
                self.add_event(
                    "WARN",
                    "cmd_odom_mismatch",
                    "cmd_v=%.3f cmd_w=%.3f odom_v=%.3f odom_w=%.3f"
                    % (cmd.linear_x, cmd.angular_z, odom.linear_x, odom.angular_z),
                )
        else:
            self.command_mismatch_since = None

        for target, source, name in (
            (self.map_frame, self.odom_frame, "map_odom_tf"),
            (self.odom_frame, self.base_frame, "odom_base_tf"),
            (self.map_frame, self.base_frame, "map_base_tf"),
        ):
            status, _age = self.lookup_tf_age(target, source, now)
            if status != "ok":
                self.add_event("WARN", name, status)

    def current_level(self, now: float) -> str:
        level = "OK"
        with self.lock:
            recent = [e for e in self.recent_events if now - e.stamp <= self.event_hold_time]
            global_pose = self.global_pose
        if any(e.level == "ERROR" for e in recent):
            level = "ERROR"
        elif any(e.level == "WARN" for e in recent):
            level = "WARN"

        if global_pose is None or now - global_pose.received > self.topic_timeout:
            level = "WARN" if level == "OK" else level
        return level

    def get_latest_event(self, now: float) -> str:
        with self.lock:
            for event in reversed(self.recent_events):
                if now - event.stamp <= self.event_hold_time:
                    return "%s:%s" % (event.event, event.detail)
        return "none"

    def format_event(self, event: EventSample) -> str:
        return "level=%s event=%s detail=%s" % (event.level, event.event, event.detail)

    def build_snapshot(self) -> Dict[str, object]:
        now = rospy.Time.now().to_sec()
        self.check_periodic_events(now)

        with self.lock:
            global_pose = self.global_pose
            rtk_pose = self.rtk_pose
            lidar_pose = self.lidar_pose
            cmd_nav = self.cmd_vel_nav
            cmd = self.cmd_vel
            odom = self.chassis_twist
            waypoint = self.waypoint_status
            fix = self.rtk_fix_type.text if self.rtk_fix_type is not None else "unknown"
            position = (
                self.rtk_position_type.text if self.rtk_position_type is not None else "unknown"
            )
            source = self.active_source
            move_base_status = self.move_base_status
            global_plan = self.global_plan
            local_plan = self.local_plan
            rpp = self.rpp_status

        rtk_lidar_xy, rtk_lidar_yaw = self.compute_rtk_lidar_delta()
        route_error = parse_float(waypoint.fields.get("route_error")) if waypoint else None
        checkpoint_distance = (
            parse_float(waypoint.fields.get("checkpoint_distance")) if waypoint else None
        )
        waypoint_state = waypoint.fields.get("state", "unknown") if waypoint else "unknown"
        waypoint_index = parse_patrol_index(waypoint.text, "checkpoint") if waypoint else None
        rpp_reason = rpp.fields.get("reason", "unknown") if rpp else "unknown"
        local_controller = rpp.fields.get("controller", "unknown") if rpp else "unknown"
        rpp_lookahead = parse_float(rpp.fields.get("lookahead")) if rpp else None
        rpp_target_x = parse_float(rpp.fields.get("target_x")) if rpp else None
        rpp_target_y = parse_float(rpp.fields.get("target_y")) if rpp else None
        rpp_curvature = parse_float(rpp.fields.get("curvature")) if rpp else None
        rpp_linear = parse_float(rpp.fields.get("linear")) if rpp else None
        rpp_angular = parse_float(rpp.fields.get("angular")) if rpp else None
        rpp_remaining = parse_float(rpp.fields.get("remaining")) if rpp else None

        tf_results = {}
        for target, source_frame, name in (
            (self.map_frame, self.odom_frame, "map_odom_tf_age"),
            (self.odom_frame, self.base_frame, "odom_base_tf_age"),
            (self.map_frame, self.base_frame, "map_base_tf_age"),
        ):
            status, age = self.lookup_tf_age(target, source_frame, now)
            tf_results[name] = age if status == "ok" else status

        level = self.current_level(now)
        last_event = self.get_latest_event(now)

        summary = (
            "source=%s rtk=%s/%s delta_xy=%s delta_yaw_deg=%s wp=%s "
            "route_error=%s controller=%s rpp_reason=%s"
            % (
                source,
                fix,
                position,
                self.format_optional(rtk_lidar_xy, "%.2f"),
                self.format_optional(
                    math.degrees(rtk_lidar_yaw) if rtk_lidar_yaw is not None else None,
                    "%.1f",
                ),
                waypoint_index if waypoint_index is not None else "unknown",
                self.format_optional(route_error, "%.2f"),
                local_controller,
                rpp_reason,
            )
        )

        return {
            "time": now,
            "level": level,
            "last_event": last_event,
            "active_source": source,
            "rtk_fix_type": fix,
            "rtk_position_type": position,
            "rtk_lidar_xy": self.format_optional(rtk_lidar_xy, "%.3f"),
            "rtk_lidar_yaw_deg": self.format_optional(
                math.degrees(rtk_lidar_yaw) if rtk_lidar_yaw is not None else None,
                "%.2f",
            ),
            "global_pose_age": self.format_optional(self.age(now, global_pose), "%.3f"),
            "rtk_pose_age": self.format_optional(self.age(now, rtk_pose), "%.3f"),
            "lidar_pose_age": self.format_optional(self.age(now, lidar_pose), "%.3f"),
            "cmd_nav_v": self.format_optional(cmd_nav.linear_x if cmd_nav else None, "%.3f"),
            "cmd_nav_w": self.format_optional(cmd_nav.angular_z if cmd_nav else None, "%.3f"),
            "cmd_v": self.format_optional(cmd.linear_x if cmd else None, "%.3f"),
            "cmd_w": self.format_optional(cmd.angular_z if cmd else None, "%.3f"),
            "odom_v": self.format_optional(odom.linear_x if odom else None, "%.3f"),
            "odom_w": self.format_optional(odom.angular_z if odom else None, "%.3f"),
            "waypoint_state": waypoint_state,
            "waypoint_index": waypoint_index if waypoint_index is not None else "unknown",
            "route_error": self.format_optional(route_error, "%.3f"),
            "checkpoint_distance": self.format_optional(checkpoint_distance, "%.3f"),
            "move_base_status": move_base_status,
            "local_controller": local_controller,
            "rpp_reason": rpp_reason,
            "rpp_status_age": self.format_optional(self.age(now, rpp), "%.3f"),
            "rpp_lookahead": self.format_optional(rpp_lookahead, "%.3f"),
            "rpp_target_x": self.format_optional(rpp_target_x, "%.3f"),
            "rpp_target_y": self.format_optional(rpp_target_y, "%.3f"),
            "rpp_curvature": self.format_optional(rpp_curvature, "%.4f"),
            "rpp_linear": self.format_optional(rpp_linear, "%.3f"),
            "rpp_angular": self.format_optional(rpp_angular, "%.3f"),
            "rpp_remaining": self.format_optional(rpp_remaining, "%.3f"),
            "global_plan_size": global_plan.size if global_plan else "unknown",
            "global_plan_length": self.format_optional(
                global_plan.length if global_plan else None, "%.3f"
            ),
            "local_plan_size": local_plan.size if local_plan else "unknown",
            "local_plan_length": self.format_optional(
                local_plan.length if local_plan else None, "%.3f"
            ),
            "map_odom_tf_age": tf_results["map_odom_tf_age"],
            "odom_base_tf_age": tf_results["odom_base_tf_age"],
            "map_base_tf_age": tf_results["map_base_tf_age"],
            "summary": summary,
        }

    def age(self, now: float, sample) -> Optional[float]:
        if sample is None:
            return None
        return now - sample.received

    def format_optional(self, value: Optional[float], fmt: str) -> str:
        if value is None:
            return "unknown"
        return fmt % value

    def status_timer_callback(self, _event) -> None:
        snapshot = self.build_snapshot()
        text = self.snapshot_to_status(snapshot)
        self.status_pub.publish(String(data=text))
        self.marker_pub.publish(self.make_markers(snapshot))

    def snapshot_timer_callback(self, _event) -> None:
        snapshot = self.build_snapshot()
        if self.snapshot_writer is not None and self.snapshot_file is not None:
            self.snapshot_writer.writerow(snapshot)
            self.snapshot_file.flush()

    def snapshot_to_status(self, snapshot: Dict[str, object]) -> str:
        return (
            "level={level} active={active_source} rtk_fix={rtk_fix_type} "
            "rtk_position={rtk_position_type} rtk_lidar_xy={rtk_lidar_xy} "
            "rtk_lidar_yaw_deg={rtk_lidar_yaw_deg} cmd_v={cmd_v} cmd_w={cmd_w} "
            "odom_v={odom_v} odom_w={odom_w} waypoint={waypoint_index} "
            "route_error={route_error} move_base={move_base_status} "
            "controller={local_controller} rpp_reason={rpp_reason} last_event={last_event}"
        ).format(**snapshot)

    def make_markers(self, snapshot: Dict[str, object]) -> MarkerArray:
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.marker_frame
        marker.ns = "nav_health"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = Point(0.0, 0.0, 1.8)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.28
        marker.color.a = 1.0
        if snapshot["level"] == "ERROR":
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
        elif snapshot["level"] == "WARN":
            marker.color.r = 1.0
            marker.color.g = 0.75
            marker.color.b = 0.05
        else:
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.2

        marker.text = (
            "NAV HEALTH: {level}\n"
            "source: {active_source} | move_base: {move_base_status}\n"
            "rtk: {rtk_fix_type}/{rtk_position_type}\n"
            "rtk-lidar: {rtk_lidar_xy} m, {rtk_lidar_yaw_deg} deg\n"
            "cmd: v={cmd_v}, w={cmd_w} | odom: v={odom_v}, w={odom_w}\n"
            "waypoint: {waypoint_index} | route_error: {route_error} m\n"
            "event: {last_event}"
        ).format(**snapshot)
        if len(marker.text) > 500:
            marker.text = marker.text[:497] + "..."
        return MarkerArray(markers=[marker])


def main() -> int:
    rospy.init_node("nav_health_monitor")
    NavHealthMonitor()
    rospy.spin()
    return 0


if __name__ == "__main__":
    main()

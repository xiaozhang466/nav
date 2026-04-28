#!/usr/bin/env python3
"""Continue a waypoint patrol from the robot's current route progress."""

import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import actionlib
import rospy
import tf
import yaml
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


@dataclass
class Waypoint:
    waypoint_id: int
    x: float
    y: float
    z: float
    yaw: float
    distance: float


def yaw_to_quaternion(yaw: float):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def goal_status_name(status: int) -> str:
    names = {
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
    return names.get(status, f"UNKNOWN({status})")


def route_distances(waypoints: Sequence[Waypoint]) -> List[float]:
    distances = []
    total = 0.0
    for index, waypoint in enumerate(waypoints):
        if index == 0:
            distances.append(0.0)
            continue
        previous = waypoints[index - 1]
        total += math.hypot(waypoint.x - previous.x, waypoint.y - previous.y)
        distances.append(total)
    return distances


def load_waypoints(path: str, frame_override: str = "") -> Tuple[str, bool, List[Waypoint]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("waypoint file must be a YAML mapping")

    frame_id = frame_override or str(data.get("frame_id", "map"))
    loop = bool(data.get("loop", False))
    raw_waypoints = data.get("waypoints", [])
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("waypoint file must contain a non-empty 'waypoints' list")

    waypoints = []
    fallback_distances = None
    if any("distance" not in item for item in raw_waypoints if isinstance(item, dict)):
        points = []
        for item in raw_waypoints:
            points.append(
                Waypoint(
                    waypoint_id=int(item.get("id", len(points))),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", 0.0)),
                    yaw=float(item.get("yaw", 0.0)),
                    distance=0.0,
                )
            )
        fallback_distances = route_distances(points)

    for index, item in enumerate(raw_waypoints):
        if not isinstance(item, dict):
            raise ValueError(f"waypoint {index} is not a mapping")
        try:
            distance = (
                float(item["distance"])
                if "distance" in item
                else float(fallback_distances[index])
            )
            waypoints.append(
                Waypoint(
                    waypoint_id=int(item.get("id", index)),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", 0.0)),
                    yaw=float(item.get("yaw", 0.0)),
                    distance=distance,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"waypoint {index} is missing numeric x/y/yaw fields") from exc

    return frame_id, loop, waypoints


def parse_active_source(status_text: str) -> str:
    """Extract the active=<source> field from a global_localization/status string."""
    for token in status_text.split():
        if token.startswith("active="):
            return token[len("active="):]
    return ""


class WaypointPatrolNode:
    def __init__(self):
        self.waypoints_file = rospy.get_param("~waypoints_file")
        frame_override = rospy.get_param("~frame_id", "")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.auto_start = bool(rospy.get_param("~auto_start", False))
        self.auto_start_retry_period = float(rospy.get_param("~auto_start_retry_period", 2.0))
        self.auto_start_timeout = float(rospy.get_param("~auto_start_timeout", 0.0))
        self.auto_start_settle_time = float(rospy.get_param("~auto_start_settle_time", 5.0))
        self.auto_start_pose_jump_tolerance = float(
            rospy.get_param("~auto_start_pose_jump_tolerance", 0.5)
        )
        self.auto_start_yaw_jump_tolerance = math.radians(
            float(rospy.get_param("~auto_start_yaw_jump_tolerance_deg", 20.0))
        )
        self.loop = bool(rospy.get_param("~loop", False))
        self.ping_pong = bool(rospy.get_param("~ping_pong", True))
        self.start_advance_distance = float(rospy.get_param("~start_advance_distance", 1.0))
        self.skip_distance = float(rospy.get_param("~skip_distance", 0.8))
        self.off_route_warn_distance = float(rospy.get_param("~off_route_warn_distance", 3.0))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 0.0))
        self.wait_for_server_timeout = float(rospy.get_param("~wait_for_server_timeout", 10.0))
        self.max_retries = int(rospy.get_param("~max_retries", 1))
        self.next_goal_delay = float(rospy.get_param("~next_goal_delay", 0.3))
        self.advance_on_pass_radius = bool(rospy.get_param("~advance_on_pass_radius", True))
        self.pass_radius = float(rospy.get_param("~pass_radius", self.skip_distance))
        self.final_pass_radius = float(rospy.get_param("~final_pass_radius", self.pass_radius))
        self.continuous_advance = bool(rospy.get_param("~continuous_advance", True))
        self.advance_distance = float(rospy.get_param("~advance_distance", 1.2))
        self.advance_check_rate = float(rospy.get_param("~advance_check_rate", 5.0))
        self.use_interpolated_goal = bool(rospy.get_param("~use_interpolated_goal", True))
        self.goal_lookahead_distance = float(rospy.get_param("~goal_lookahead_distance", 4.0))
        self.goal_update_distance = float(rospy.get_param("~goal_update_distance", 0.75))
        self.goal_update_min_interval = float(rospy.get_param("~goal_update_min_interval", 0.7))
        self.route_realign_distance = float(rospy.get_param("~route_realign_distance", 2.0))
        self.passed_checkpoint_realign_distance = float(
            rospy.get_param("~passed_checkpoint_realign_distance", 1.5)
        )
        self.passed_checkpoint_realign_route_error = float(
            rospy.get_param("~passed_checkpoint_realign_route_error", 2.0)
        )
        self.realign_while_running = bool(rospy.get_param("~realign_while_running", True))
        self.rejoin_mode_enabled = bool(rospy.get_param("~rejoin_mode_enabled", True))
        self.rejoin_enter_route_error = float(
            rospy.get_param(
                "~rejoin_enter_route_error",
                max(self.off_route_warn_distance, self.pass_radius * 2.0),
            )
        )
        self.rejoin_exit_route_error = float(
            rospy.get_param(
                "~rejoin_exit_route_error",
                max(self.pass_radius, 0.8),
            )
        )
        self.rejoin_goal_lookahead_distance = float(
            rospy.get_param("~rejoin_goal_lookahead_distance", 0.0)
        )
        self.rejoin_goal_update_distance = float(
            rospy.get_param("~rejoin_goal_update_distance", 0.3)
        )
        self.rejoin_use_route_yaw = bool(rospy.get_param("~rejoin_use_route_yaw", False))
        self.startup_mode_enabled = bool(rospy.get_param("~startup_mode_enabled", True))
        self.startup_goal_lookahead_distance = float(
            rospy.get_param(
                "~startup_goal_lookahead_distance",
                min(2.0, max(0.5, self.goal_lookahead_distance)),
            )
        )
        self.startup_enter_route_error = float(
            rospy.get_param("~startup_enter_route_error", max(0.35, self.pass_radius))
        )
        self.startup_exit_route_error = float(
            rospy.get_param("~startup_exit_route_error", max(0.20, self.pass_radius * 0.5))
        )
        self.startup_enter_yaw_error = math.radians(
            float(rospy.get_param("~startup_enter_yaw_error_deg", 30.0))
        )
        self.startup_exit_yaw_error = math.radians(
            float(rospy.get_param("~startup_exit_yaw_error_deg", 15.0))
        )
        self.startup_release_progress = float(
            rospy.get_param(
                "~startup_release_progress",
                max(self.startup_goal_lookahead_distance, self.start_advance_distance + 1.0),
            )
        )
        self.startup_use_route_yaw = bool(rospy.get_param("~startup_use_route_yaw", True))
        self.terminal_mode_enabled = bool(rospy.get_param("~terminal_mode_enabled", True))
        self.terminal_mode_distance = float(
            rospy.get_param("~terminal_mode_distance", max(6.0, self.goal_lookahead_distance))
        )
        self.terminal_pass_radius = float(
            rospy.get_param("~terminal_pass_radius", max(self.pass_radius, self.final_pass_radius))
        )
        self.terminal_progress_tolerance = float(
            rospy.get_param("~terminal_progress_tolerance", self.skip_distance)
        )
        self.terminal_use_waypoint_yaw = bool(
            rospy.get_param("~terminal_use_waypoint_yaw", True)
        )
        self.goal_lookahead_count = max(1, int(rospy.get_param("~goal_lookahead_count", 2)))
        self.source_switch_suppress_duration = float(
            rospy.get_param("~source_switch_suppress_duration", 5.0)
        )
        self.global_status_topic = rospy.get_param(
            "~global_status_topic", "/global_localization/status"
        )
        self.status_rate = float(rospy.get_param("~status_rate", 1.0))
        self.use_waypoint_z = bool(rospy.get_param("~use_waypoint_z", False))
        self.goal_z = float(rospy.get_param("~goal_z", 0.0))
        self.ignore_goal_yaw = bool(rospy.get_param("~ignore_goal_yaw", True))
        self.turn_in_place_enabled = bool(rospy.get_param("~turn_in_place_enabled", True))
        self.turn_cmd_vel_topic = rospy.get_param("~turn_cmd_vel_topic", "/cmd_vel_nav")
        self.turn_control_rate = float(rospy.get_param("~turn_control_rate", 10.0))
        self.turn_yaw_tolerance = math.radians(
            float(rospy.get_param("~turn_yaw_tolerance_deg", 8.0))
        )
        self.turn_angular_speed = float(rospy.get_param("~turn_angular_speed", 0.35))
        self.turn_min_angular_speed = float(rospy.get_param("~turn_min_angular_speed", 0.12))
        self.turn_kp = float(rospy.get_param("~turn_kp", 1.2))
        self.turn_timeout = float(rospy.get_param("~turn_timeout", 14.0))

        if not self.waypoints_file:
            raise ValueError("~waypoints_file is required")

        self.frame_id, file_loop, self.waypoints = load_waypoints(
            self.waypoints_file, frame_override=frame_override
        )
        self.loop = self.loop or file_loop

        self.tf_listener = tf.TransformListener()
        self.move_base = actionlib.SimpleActionClient(self.move_base_action, MoveBaseAction)

        self.lock = threading.RLock()
        self.state = "IDLE"
        self.current_index: Optional[int] = None
        self.wrapping_to_start = False
        self.active_goal_index: Optional[int] = None
        self.active_goal_distance: Optional[float] = None
        self.active_goal_x: Optional[float] = None
        self.active_goal_y: Optional[float] = None
        self.active_goal_terminal = False
        self.active_goal_sequence = 0
        self.last_goal_sent_time = rospy.Time(0)
        self.current_retries = 0
        self.route_direction = 1
        self.last_message = "loaded"
        self.last_nearest_distance: Optional[float] = None
        self.last_route_progress: Optional[float] = None
        self.last_checkpoint_distance: Optional[float] = None
        self.last_terminal_mode = False
        self.last_passed_checkpoint_index: Optional[int] = None
        self.last_passed_checkpoint_progress: Optional[float] = None
        self.last_passed_checkpoint_direction = 0
        self.rejoin_mode_active = False
        self.startup_mode_active = False
        self.startup_mode_start_progress: Optional[float] = None
        self.waiting_for_server = True
        self.auto_start_begin_time: Optional[rospy.Time] = None
        self.auto_start_candidate_pose: Optional[Tuple[float, float, float]] = None
        self.auto_start_candidate_time: Optional[rospy.Time] = None
        self.turn_target_yaw: Optional[float] = None
        self.turn_started_at: Optional[rospy.Time] = None
        self.last_known_active_source: Optional[str] = None
        self.source_switch_suppress_until: Optional[float] = None

        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.current_goal_pub = rospy.Publisher("~current_goal", PoseStamped, queue_size=1, latch=True)
        self.turn_cmd_pub = rospy.Publisher(self.turn_cmd_vel_topic, Twist, queue_size=1)

        rospy.Service("~start", Trigger, self.handle_start)
        rospy.Service("~pause", Trigger, self.handle_pause)
        rospy.Service("~resume", Trigger, self.handle_resume)
        rospy.Service("~stop", Trigger, self.handle_stop)

        period = 1.0 / max(self.status_rate, 0.1)
        self.status_timer = rospy.Timer(rospy.Duration(period), lambda _event: self.publish_status())
        advance_period = 1.0 / max(self.advance_check_rate, 0.5)
        self.advance_timer = rospy.Timer(
            rospy.Duration(advance_period), self.handle_continuous_advance
        )
        turn_period = 1.0 / max(self.turn_control_rate, 1.0)
        self.turn_timer = rospy.Timer(rospy.Duration(turn_period), self.handle_turning)
        rospy.Subscriber(
            self.global_status_topic, String, self._global_status_callback, queue_size=5
        )

        rospy.loginfo(
            "Loaded %d patrol waypoints from %s in frame %s",
            len(self.waypoints),
            os.path.abspath(self.waypoints_file),
            self.frame_id,
        )

        wait_chunk = max(0.5, self.wait_for_server_timeout)
        while not rospy.is_shutdown():
            if self.move_base.wait_for_server(rospy.Duration(wait_chunk)):
                break
            rospy.logwarn(
                "Waiting for move_base action server %s to become available...",
                self.move_base_action,
            )
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("shutdown while waiting for move_base action server")
        self.waiting_for_server = False
        rospy.loginfo("Connected to move_base action server %s", self.move_base_action)

        if self.auto_start:
            self.auto_start_begin_time = rospy.Time.now()
            rospy.Timer(rospy.Duration(0.5), self._auto_start_once, oneshot=True)

    def auto_start_pose_is_settled(self) -> Tuple[bool, str]:
        if self.auto_start_settle_time <= 0.0:
            return True, "auto-start settle disabled"

        try:
            x, y, yaw = self.get_current_xy_yaw()
        except Exception as exc:  # pylint: disable=broad-except
            self.auto_start_candidate_pose = None
            self.auto_start_candidate_time = None
            return False, f"waiting for localization TF: {exc}"

        now = rospy.Time.now()
        current_pose = (x, y, yaw)
        if self.auto_start_candidate_pose is None or self.auto_start_candidate_time is None:
            self.auto_start_candidate_pose = current_pose
            self.auto_start_candidate_time = now
            return (
                False,
                "waiting for localization pose to settle: 0.0/%.1fs"
                % self.auto_start_settle_time,
            )

        start_x, start_y, start_yaw = self.auto_start_candidate_pose
        xy_jump = math.hypot(x - start_x, y - start_y)
        yaw_jump = abs(wrap_angle(yaw - start_yaw))
        if (
            xy_jump > self.auto_start_pose_jump_tolerance
            or yaw_jump > self.auto_start_yaw_jump_tolerance
        ):
            self.auto_start_candidate_pose = current_pose
            self.auto_start_candidate_time = now
            return (
                False,
                "localization pose jumped during auto-start settle: "
                "xy=%.2fm yaw=%.1fdeg, restarting settle timer"
                % (xy_jump, math.degrees(yaw_jump)),
            )

        elapsed = (now - self.auto_start_candidate_time).to_sec()
        if elapsed < self.auto_start_settle_time:
            return (
                False,
                "waiting for localization pose to settle: %.1f/%.1fs"
                % (elapsed, self.auto_start_settle_time),
            )

        return True, "localization pose settled for %.1fs" % elapsed

    def _auto_start_once(self, _event):
        settled, settle_message = self.auto_start_pose_is_settled()
        if not settled:
            self._schedule_auto_start_retry(settle_message)
            return

        response = self.start_from_current_pose()
        if response.success:
            rospy.loginfo("Auto-start patrol succeeded: %s", response.message)
            return

        if self.state in ("RUNNING", "ADVANCING", "RETRYING"):
            rospy.loginfo("Auto-start patrol stopped retrying because patrol is already running")
            return

        self._schedule_auto_start_retry(response.message)

    def _schedule_auto_start_retry(self, message: str):
        now = rospy.Time.now()
        elapsed = (
            (now - self.auto_start_begin_time).to_sec()
            if self.auto_start_begin_time is not None
            else 0.0
        )
        if self.auto_start_timeout > 0.0 and elapsed >= self.auto_start_timeout:
            rospy.logerr(
                "Auto-start patrol timed out after %.1fs: %s",
                elapsed,
                message,
            )
            return

        rospy.logwarn(
            "Auto-start patrol failed, retrying in %.1fs: %s",
            self.auto_start_retry_period,
            message,
        )
        rospy.Timer(
            rospy.Duration(max(0.1, self.auto_start_retry_period)),
            self._auto_start_once,
            oneshot=True,
        )

    def get_current_xy(self) -> Tuple[float, float]:
        self.tf_listener.waitForTransform(
            self.frame_id, self.base_frame, rospy.Time(0), rospy.Duration(2.0)
        )
        translation, _rotation = self.tf_listener.lookupTransform(
            self.frame_id, self.base_frame, rospy.Time(0)
        )
        return float(translation[0]), float(translation[1])

    def get_current_xy_yaw(self) -> Tuple[float, float, float]:
        self.tf_listener.waitForTransform(
            self.frame_id, self.base_frame, rospy.Time(0), rospy.Duration(2.0)
        )
        translation, rotation = self.tf_listener.lookupTransform(
            self.frame_id, self.base_frame, rospy.Time(0)
        )
        yaw = tf.transformations.euler_from_quaternion(rotation)[2]
        return float(translation[0]), float(translation[1]), float(yaw)

    def refresh_route_state_locked(self) -> Tuple[float, float]:
        x, y = self.get_current_xy()
        progress, nearest_distance = self.closest_route_progress(x, y)
        self.last_route_progress = progress
        self.last_nearest_distance = nearest_distance
        return progress, nearest_distance

    def closest_route_progress(self, x: float, y: float) -> Tuple[float, float]:
        if len(self.waypoints) == 1:
            waypoint = self.waypoints[0]
            return waypoint.distance, math.hypot(x - waypoint.x, y - waypoint.y)

        best_progress = self.waypoints[0].distance
        best_distance = float("inf")
        for first, second in zip(self.waypoints, self.waypoints[1:]):
            dx = second.x - first.x
            dy = second.y - first.y
            segment_sq = dx * dx + dy * dy
            if segment_sq <= 1e-9:
                ratio = 0.0
            else:
                ratio = ((x - first.x) * dx + (y - first.y) * dy) / segment_sq
                ratio = max(0.0, min(1.0, ratio))
            proj_x = first.x + dx * ratio
            proj_y = first.y + dy * ratio
            dist = math.hypot(x - proj_x, y - proj_y)
            progress = first.distance + (second.distance - first.distance) * ratio
            if dist < best_distance:
                best_distance = dist
                best_progress = progress
        return best_progress, best_distance

    def route_pose_at_progress(self, target_progress: float) -> Waypoint:
        if target_progress <= self.waypoints[0].distance:
            first = self.waypoints[0]
            yaw = first.yaw
            if len(self.waypoints) > 1:
                second = self.waypoints[1]
                yaw = math.atan2(second.y - first.y, second.x - first.x)
            return Waypoint(-1, first.x, first.y, first.z, yaw, first.distance)

        last = self.waypoints[-1]
        if target_progress >= last.distance:
            yaw = last.yaw
            if len(self.waypoints) > 1:
                previous = self.waypoints[-2]
                yaw = math.atan2(last.y - previous.y, last.x - previous.x)
            return Waypoint(-1, last.x, last.y, last.z, yaw, last.distance)

        for first, second in zip(self.waypoints, self.waypoints[1:]):
            if second.distance < target_progress:
                continue
            segment_distance = max(1e-9, second.distance - first.distance)
            ratio = (target_progress - first.distance) / segment_distance
            yaw = math.atan2(second.y - first.y, second.x - first.x)
            return Waypoint(
                waypoint_id=-1,
                x=first.x + (second.x - first.x) * ratio,
                y=first.y + (second.y - first.y) * ratio,
                z=first.z + (second.z - first.z) * ratio,
                yaw=yaw,
                distance=target_progress,
            )

        return last

    def route_yaw_for_direction(self, route_progress: float) -> float:
        yaw = self.route_pose_at_progress(route_progress).yaw
        if self.route_direction < 0:
            yaw = wrap_angle(yaw + math.pi)
        return yaw

    def route_pose_at_progress_for_direction(self, target_progress: float) -> Waypoint:
        pose = self.route_pose_at_progress(target_progress)
        if self.route_direction < 0:
            pose.yaw = wrap_angle(pose.yaw + math.pi)
        return pose

    def terminal_index_for_direction(self) -> int:
        return len(self.waypoints) - 1 if self.route_direction > 0 else 0

    def terminal_progress_for_direction(self) -> float:
        return self.waypoints[self.terminal_index_for_direction()].distance

    def terminal_remaining_distance(self, route_progress: float) -> float:
        if self.route_direction > 0:
            return max(0.0, self.waypoints[-1].distance - route_progress)
        return max(0.0, route_progress - self.waypoints[0].distance)

    def next_index_after_turn(self) -> int:
        if len(self.waypoints) <= 1:
            return 0
        if self.route_direction > 0:
            return 1
        return len(self.waypoints) - 2

    def target_goal_distance(self, route_progress: float, checkpoint_index: int) -> float:
        final_distance = self.waypoints[-1].distance
        lookahead_distance = self.goal_lookahead_distance
        if self.startup_mode_active:
            lookahead_distance = min(lookahead_distance, self.startup_goal_lookahead_distance)
        if self.loop and not self.ping_pong and final_distance > 0.0:
            return (route_progress + lookahead_distance) % final_distance
        if self.route_direction > 0:
            return min(final_distance, route_progress + lookahead_distance)
        return max(self.waypoints[0].distance, route_progress - lookahead_distance)

    def rejoin_goal_distance(self, route_progress: float) -> float:
        final_distance = self.waypoints[-1].distance
        lookahead_distance = max(0.0, self.rejoin_goal_lookahead_distance)
        if self.loop and not self.ping_pong and final_distance > 0.0:
            return (route_progress + lookahead_distance) % final_distance
        if self.route_direction > 0:
            return min(final_distance, route_progress + lookahead_distance)
        return max(self.waypoints[0].distance, route_progress - lookahead_distance)

    def heading_error_to_route(self, route_progress: float, current_yaw: float) -> float:
        route_yaw = self.route_yaw_for_direction(route_progress)
        return abs(wrap_angle(route_yaw - current_yaw))

    def evaluate_rejoin_mode(self, nearest_distance: float) -> bool:
        return (
            self.rejoin_mode_enabled
            and self.use_interpolated_goal
            and nearest_distance >= self.rejoin_enter_route_error
        )

    def evaluate_startup_mode(
        self, route_progress: float, nearest_distance: float, current_yaw: float
    ) -> Tuple[bool, float]:
        yaw_error = self.heading_error_to_route(route_progress, current_yaw)
        should_enable = (
            self.startup_mode_enabled
            and self.use_interpolated_goal
            and (
                nearest_distance >= self.startup_enter_route_error
                or yaw_error >= self.startup_enter_yaw_error
            )
        )
        return should_enable, yaw_error

    def _source_switch_suppressed(self) -> bool:
        """Return True if startup mode entry is suppressed due to a recent
        localization source switch (to avoid false yaw-error triggers)."""
        if self.source_switch_suppress_until is None:
            return False
        return rospy.Time.now().to_sec() < self.source_switch_suppress_until

    def _global_status_callback(self, msg: String) -> None:
        """Detect localization source switches and suppress startup mode."""
        active = parse_active_source(msg.data)
        if not active or active == "none":
            return
        with self.lock:
            previous = self.last_known_active_source
            self.last_known_active_source = active
            if previous is not None and previous != active:
                self.clear_passed_checkpoint_guard_locked()
                suppress_until = (
                    rospy.Time.now().to_sec() + self.source_switch_suppress_duration
                )
                self.source_switch_suppress_until = suppress_until
                rospy.loginfo(
                    "Localization source switched %s -> %s, suppressing "
                    "startup mode for %.1fs",
                    previous,
                    active,
                    self.source_switch_suppress_duration,
                )

    def maybe_enter_startup_mode_locked(
        self, route_progress: float, nearest_distance: float, current_yaw: float
    ) -> bool:
        if self.startup_mode_active:
            return False

        if self._source_switch_suppressed():
            return False

        should_enable, yaw_error = self.evaluate_startup_mode(
            route_progress, nearest_distance, current_yaw
        )
        if not should_enable:
            return False

        self.startup_mode_active = True
        self.startup_mode_start_progress = route_progress
        self.current_retries = 0
        rospy.loginfo(
            "Entering startup patrol mode: route_error=%.2fm yaw_error=%.1fdeg lookahead=%.2fm",
            nearest_distance,
            math.degrees(yaw_error),
            self.startup_goal_lookahead_distance,
        )
        return True

    def maybe_update_startup_mode_locked(
        self,
        route_progress: Optional[float] = None,
        nearest_distance: Optional[float] = None,
    ) -> bool:
        if not self.startup_mode_active:
            return False

        if route_progress is None or nearest_distance is None:
            x, y, current_yaw = self.get_current_xy_yaw()
            route_progress, nearest_distance = self.closest_route_progress(x, y)
        else:
            _x, _y, current_yaw = self.get_current_xy_yaw()
        yaw_error = self.heading_error_to_route(route_progress, current_yaw)
        if self.startup_mode_start_progress is None:
            progress_delta = 0.0
        elif self.route_direction > 0:
            progress_delta = max(0.0, route_progress - self.startup_mode_start_progress)
        else:
            progress_delta = max(0.0, self.startup_mode_start_progress - route_progress)
        if (
            progress_delta >= self.startup_release_progress
            or (
                nearest_distance <= self.startup_exit_route_error
                and yaw_error <= self.startup_exit_yaw_error
            )
        ):
            rospy.loginfo(
                "Leaving startup patrol mode: route_error=%.2fm yaw_error=%.1fdeg progress_delta=%.2fm",
                nearest_distance,
                math.degrees(yaw_error),
                progress_delta,
            )
            self.startup_mode_active = False
            self.startup_mode_start_progress = None
            return True
        return False

    def maybe_update_rejoin_mode_locked(
        self,
        route_progress: Optional[float] = None,
        nearest_distance: Optional[float] = None,
    ) -> bool:
        if not self.rejoin_mode_active:
            return False

        if route_progress is None or nearest_distance is None:
            route_progress, nearest_distance = self.refresh_route_state_locked()
        else:
            self.last_route_progress = route_progress
            self.last_nearest_distance = nearest_distance

        if nearest_distance <= self.rejoin_exit_route_error:
            rospy.loginfo(
                "Leaving route rejoin mode: route_error=%.2fm progress=%.2fm",
                nearest_distance,
                route_progress,
            )
            self.rejoin_mode_active = False
            return True
        return False

    def remaining_route_distance(self, route_progress: float) -> float:
        return max(0.0, self.waypoints[-1].distance - route_progress)

    def in_terminal_mode(self, route_progress: float, checkpoint_index: Optional[int]) -> bool:
        if not self.terminal_mode_enabled or not self.use_interpolated_goal:
            return False
        if len(self.waypoints) < 2:
            return False
        terminal_index = self.terminal_index_for_direction()
        if checkpoint_index is not None and checkpoint_index == terminal_index:
            return True
        return self.terminal_remaining_distance(route_progress) <= self.terminal_mode_distance

    def remember_passed_checkpoint_locked(
        self, passed_index: int, route_progress: float
    ) -> None:
        self.last_passed_checkpoint_index = passed_index
        self.last_passed_checkpoint_progress = route_progress
        self.last_passed_checkpoint_direction = self.route_direction

    def clear_passed_checkpoint_guard_locked(self) -> None:
        self.last_passed_checkpoint_index = None
        self.last_passed_checkpoint_progress = None
        self.last_passed_checkpoint_direction = 0

    def should_suppress_realign_to_passed_checkpoint_locked(
        self,
        desired_index: int,
        route_progress: float,
        nearest_distance: float,
    ) -> bool:
        if self.last_passed_checkpoint_index is None:
            return False
        if self.last_passed_checkpoint_progress is None:
            return False
        if desired_index != self.last_passed_checkpoint_index:
            return False
        if self.last_passed_checkpoint_direction != self.route_direction:
            self.clear_passed_checkpoint_guard_locked()
            return False
        if nearest_distance >= self.passed_checkpoint_realign_route_error:
            self.clear_passed_checkpoint_guard_locked()
            return False

        progress_after_pass = (
            (route_progress - self.last_passed_checkpoint_progress) * self.route_direction
        )
        if progress_after_pass >= self.passed_checkpoint_realign_distance:
            self.clear_passed_checkpoint_guard_locked()
            return False

        rospy.logwarn_throttle(
            2.0,
            (
                "Suppressing checkpoint realign back to recently passed index=%d: "
                "progress_after_pass=%.2fm route_error=%.2fm"
            ),
            desired_index,
            progress_after_pass,
            nearest_distance,
        )
        return True

    def try_complete_terminal_segment_locked(
        self, route_progress: float, x: float, y: float
    ) -> bool:
        if not self.in_terminal_mode(route_progress, self.current_index):
            return False

        terminal_index = self.terminal_index_for_direction()
        terminal_waypoint = self.waypoints[terminal_index]
        terminal_distance = math.hypot(terminal_waypoint.x - x, terminal_waypoint.y - y)
        remaining_distance = self.terminal_remaining_distance(route_progress)
        self.last_checkpoint_distance = terminal_distance

        if (
            terminal_distance > self.terminal_pass_radius
            and remaining_distance > self.terminal_progress_tolerance
        ):
            return False

        rospy.loginfo(
            "Completed terminal patrol segment: terminal_index=%d distance=%.2fm remaining_route=%.2fm",
            terminal_index,
            terminal_distance,
            remaining_distance,
        )
        self.current_retries = 0
        self.active_goal_index = None
        self.active_goal_distance = None
        self.active_goal_x = None
        self.active_goal_y = None
        self.active_goal_terminal = False

        if self.loop:
            if self.ping_pong:
                self.start_turnaround_locked(
                    "completed terminal segment at index=%d" % terminal_index
                )
            else:
                self.current_index = 0
                self.wrapping_to_start = True
                self.schedule_send_goal_locked(
                    "ADVANCING",
                    "completed terminal segment, scheduling loop restart at index=0",
                )
            return True

        self.current_index = len(self.waypoints)
        self.state = "FINISHED"
        self.last_message = "completed terminal segment"
        self.move_base.cancel_all_goals()
        return True

    def realign_checkpoint_from_progress_locked(
        self, route_progress: float, nearest_distance: float
    ):
        if self.in_terminal_mode(route_progress, self.current_index):
            terminal_index = self.terminal_index_for_direction()
            if self.current_index != terminal_index:
                rospy.loginfo(
                    "Entering terminal patrol mode: checkpoint index=%d -> final index=%d remaining=%.2fm",
                    -1 if self.current_index is None else self.current_index,
                    terminal_index,
                    self.terminal_remaining_distance(route_progress),
                )
            self.current_index = terminal_index
            self.last_terminal_mode = True
            return

        desired_index = self.choose_start_index(route_progress)
        if desired_index is None:
            self.current_index = len(self.waypoints)
            return
        if self.should_suppress_realign_to_passed_checkpoint_locked(
            desired_index, route_progress, nearest_distance
        ):
            self.last_terminal_mode = False
            return

        if self.current_index is None:
            self.current_index = desired_index
            return

        current_distance = self.waypoints[
            min(self.current_index, len(self.waypoints) - 1)
        ].distance
        desired_distance = self.waypoints[desired_index].distance
        if abs(current_distance - desired_distance) >= self.route_realign_distance:
            rospy.logwarn(
                "Realigning patrol checkpoint from index=%d distance=%.2f to index=%d distance=%.2f based on route_progress=%.2f",
                self.current_index,
                current_distance,
                desired_index,
                desired_distance,
                route_progress,
            )
            self.current_index = desired_index
        self.last_terminal_mode = False

    def choose_start_index(self, route_progress: float) -> Optional[int]:
        if self.route_direction > 0:
            target_progress = route_progress + self.start_advance_distance
            for index, waypoint in enumerate(self.waypoints):
                if waypoint.distance > target_progress:
                    return index
            if self.loop and self.ping_pong:
                return len(self.waypoints) - 1
            return 0 if self.loop else None

        target_progress = route_progress - self.start_advance_distance
        for index in range(len(self.waypoints) - 1, -1, -1):
            if self.waypoints[index].distance < target_progress:
                return index
        return 0 if self.loop and self.ping_pong else None

    def advance_past_nearby_waypoints(self, index: int) -> int:
        try:
            x, y = self.get_current_xy()
        except Exception:
            return index

        if self.route_direction > 0:
            while index < len(self.waypoints):
                waypoint = self.waypoints[index]
                if math.hypot(waypoint.x - x, waypoint.y - y) > self.skip_distance:
                    break
                index += 1
                if index >= len(self.waypoints) and self.loop:
                    index = len(self.waypoints) - 1 if self.ping_pong else 0
                    break
            return index

        while index >= 0:
            waypoint = self.waypoints[index]
            if math.hypot(waypoint.x - x, waypoint.y - y) > self.skip_distance:
                break
            index -= 1
            if index < 0 and self.loop:
                index = 0
                break
        return index

    def advance_past_reached_waypoints_locked(self):
        if self.current_index is None:
            return
        if self.rejoin_mode_active:
            return

        try:
            progress, nearest_distance = self.refresh_route_state_locked()
        except Exception:
            self.current_index = self.advance_past_nearby_waypoints(self.current_index)
            return

        if (
            self.loop
            and not self.ping_pong
            and self.current_index == 0
            and (
                self.wrapping_to_start
                or progress >= self.waypoints[-1].distance - self.skip_distance
            )
        ):
            # We have just passed the final checkpoint and intentionally need
            # to send waypoint 0.  The open-route progress is still near the
            # route end here, so the usual "skip passed checkpoints" logic
            # would incorrectly advance back to the final waypoint.
            self.wrapping_to_start = True
            return

        if self.use_interpolated_goal:
            self.realign_checkpoint_from_progress_locked(progress, nearest_distance)
            return

        if self.route_direction > 0:
            while self.current_index < len(self.waypoints) - 1:
                checkpoint = self.waypoints[self.current_index]
                if progress < checkpoint.distance - self.advance_distance:
                    break
                self.current_index += 1
        else:
            while self.current_index > 0:
                checkpoint = self.waypoints[self.current_index]
                if progress > checkpoint.distance + self.advance_distance:
                    break
                self.current_index -= 1
        if self.current_index != 0:
            self.wrapping_to_start = False

    def goal_index_for_checkpoint(self, checkpoint_index: int) -> int:
        if self.loop:
            if self.ping_pong:
                return max(
                    0,
                    min(
                        len(self.waypoints) - 1,
                        checkpoint_index
                        + self.route_direction * (self.goal_lookahead_count - 1),
                    ),
                )
            return (checkpoint_index + self.goal_lookahead_count - 1) % len(self.waypoints)
        if self.route_direction > 0:
            return min(len(self.waypoints) - 1, checkpoint_index + self.goal_lookahead_count - 1)
        return max(0, checkpoint_index - self.goal_lookahead_count + 1)

    def goal_for_checkpoint(self, checkpoint_index: int) -> Tuple[Waypoint, Optional[int], bool]:
        if self.use_interpolated_goal:
            try:
                route_progress, _nearest_distance = self.refresh_route_state_locked()
            except Exception:
                route_progress = self.waypoints[max(0, checkpoint_index - 1)].distance
            if self.rejoin_mode_active:
                target_distance = self.rejoin_goal_distance(route_progress)
                self.last_terminal_mode = False
                return (
                    self.route_pose_at_progress_for_direction(target_distance),
                    None,
                    self.rejoin_use_route_yaw,
                )
            if self.startup_mode_active:
                target_distance = self.target_goal_distance(route_progress, checkpoint_index)
                self.last_terminal_mode = False
                return (
                    self.route_pose_at_progress_for_direction(target_distance),
                    None,
                    self.startup_use_route_yaw,
                )
            if self.in_terminal_mode(route_progress, checkpoint_index):
                self.last_terminal_mode = True
                terminal_index = self.terminal_index_for_direction()
                return (
                    self.route_pose_at_progress_for_direction(
                        self.waypoints[terminal_index].distance
                    ),
                    terminal_index,
                    True,
                )
            target_distance = self.target_goal_distance(route_progress, checkpoint_index)
            self.last_terminal_mode = False
            return self.route_pose_at_progress_for_direction(target_distance), None, False

        goal_index = self.goal_index_for_checkpoint(checkpoint_index)
        self.last_terminal_mode = False
        return self.waypoints[goal_index], goal_index, False

    def make_goal_pose(self, waypoint: Waypoint, preserve_waypoint_yaw: bool = False) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        pose.pose.position.z = waypoint.z if self.use_waypoint_z else self.goal_z
        goal_yaw = waypoint.yaw
        if self.ignore_goal_yaw and not preserve_waypoint_yaw:
            try:
                x, y, current_yaw = self.get_current_xy_yaw()
                if math.hypot(waypoint.x - x, waypoint.y - y) > 1e-6:
                    goal_yaw = math.atan2(waypoint.y - y, waypoint.x - x)
                else:
                    goal_yaw = current_yaw
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logdebug("Keeping waypoint yaw because current yaw lookup failed: %s", exc)
        qx, qy, qz, qw = yaw_to_quaternion(goal_yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def send_goal_locked(self):
        if (
            self.current_index is None
            or self.current_index < 0
            or self.current_index >= len(self.waypoints)
        ):
            if self.loop and self.ping_pong and self.current_index is not None:
                self.start_turnaround_locked("route boundary reached")
                return
            self.state = "FINISHED"
            self.last_message = "no more waypoints"
            self.move_base.cancel_all_goals()
            return

        if not self.rejoin_mode_active:
            self.advance_past_reached_waypoints_locked()
        if self.current_index is not None and self.current_index < 0:
            if self.loop and self.ping_pong:
                self.start_turnaround_locked("route boundary reached")
                return
            self.state = "FINISHED"
            self.last_message = "finished route"
            self.move_base.cancel_all_goals()
            return
        if self.current_index >= len(self.waypoints):
            if self.loop:
                if self.ping_pong:
                    self.start_turnaround_locked("route boundary reached")
                    return
                self.current_index = 0
            else:
                self.state = "FINISHED"
                self.last_message = "finished route"
                self.move_base.cancel_all_goals()
                return

        checkpoint_index = self.current_index
        try:
            x, y, current_yaw = self.get_current_xy_yaw()
            route_progress, nearest_distance = self.closest_route_progress(x, y)
            self.last_route_progress = route_progress
            self.last_nearest_distance = nearest_distance
            if self.startup_mode_active:
                self.maybe_update_startup_mode_locked(route_progress, nearest_distance)
            else:
                self.maybe_enter_startup_mode_locked(
                    route_progress, nearest_distance, current_yaw
                )
        except Exception:
            pass
        waypoint, goal_index, terminal_goal = self.goal_for_checkpoint(checkpoint_index)
        pose = self.make_goal_pose(
            waypoint,
            preserve_waypoint_yaw=(
                (terminal_goal and self.terminal_use_waypoint_yaw)
                or (self.startup_mode_active and self.startup_use_route_yaw)
            ),
        )
        goal = MoveBaseGoal(target_pose=pose)
        self.current_goal_pub.publish(pose)
        self.active_goal_sequence += 1
        goal_sequence = self.active_goal_sequence
        self.active_goal_index = goal_index
        self.active_goal_distance = waypoint.distance
        self.active_goal_x = waypoint.x
        self.active_goal_y = waypoint.y
        self.active_goal_terminal = terminal_goal
        self.last_goal_sent_time = rospy.Time.now()
        self.state = "RUNNING"
        self.last_message = (
            f"sent goal id={waypoint.waypoint_id} index={goal_index} "
            f"checkpoint={checkpoint_index}"
        )
        rospy.loginfo(
            "Sending patrol goal id=%d x=%.3f y=%.3f yaw=%.1f deg distance=%.2f (checkpoint index=%d)",
            waypoint.waypoint_id,
            waypoint.x,
            waypoint.y,
            math.degrees(waypoint.yaw),
            waypoint.distance,
            checkpoint_index,
        )
        self.move_base.send_goal(
            goal,
            done_cb=lambda status, result: self.handle_goal_done(goal_sequence, status, result),
            active_cb=None,
            feedback_cb=None,
        )

        if self.goal_timeout > 0.0:
            timeout_sequence = goal_sequence
            rospy.Timer(
                rospy.Duration(self.goal_timeout),
                lambda event: self._goal_timeout_once(event, timeout_sequence),
                oneshot=True,
            )

    def _goal_timeout_once(self, _event, goal_sequence):
        with self.lock:
            if self.state != "RUNNING" or self.active_goal_sequence != goal_sequence:
                return
            self.state = "FAILED"
            self.last_message = f"goal timeout after {self.goal_timeout:.1f}s"
            self.move_base.cancel_all_goals()

    def schedule_send_goal_locked(self, state: str, message: str):
        self.state = state
        self.last_message = message
        scheduled_index = self.current_index
        delay = max(0.0, self.next_goal_delay)
        rospy.Timer(
            rospy.Duration(delay),
            lambda event: self._send_scheduled_goal_once(event, scheduled_index),
            oneshot=True,
        )

    def _send_scheduled_goal_once(self, _event, scheduled_index):
        with self.lock:
            if self.current_index != scheduled_index:
                return
            if self.state not in ("ADVANCING", "RETRYING"):
                return
            self.send_goal_locked()

    def publish_turn_command(self, angular_z: float) -> None:
        cmd = Twist()
        cmd.angular.z = angular_z
        self.turn_cmd_pub.publish(cmd)

    def stop_turn_command(self) -> None:
        self.publish_turn_command(0.0)

    def start_turnaround_locked(self, reason: str) -> None:
        self.move_base.cancel_all_goals()
        self.stop_turn_command()
        self.current_retries = 0
        turn_progress = self.terminal_progress_for_direction()
        self.route_direction *= -1
        self.current_index = self.next_index_after_turn()
        self.wrapping_to_start = False
        self.rejoin_mode_active = False
        self.startup_mode_active = False
        self.startup_mode_start_progress = None
        self.clear_passed_checkpoint_guard_locked()
        self.active_goal_index = None
        self.active_goal_distance = None
        self.active_goal_x = None
        self.active_goal_y = None
        self.active_goal_terminal = False

        self.turn_target_yaw = self.route_yaw_for_direction(turn_progress)
        self.turn_started_at = rospy.Time.now()

        direction_text = "forward" if self.route_direction > 0 else "reverse"
        if not self.turn_in_place_enabled:
            self.schedule_send_goal_locked(
                "ADVANCING",
                "%s, direction=%s, turn disabled" % (reason, direction_text),
            )
            return

        self.state = "TURNING"
        self.last_message = (
            "%s, turning in place to %s heading yaw=%.1fdeg"
            % (reason, direction_text, math.degrees(self.turn_target_yaw))
        )
        rospy.logwarn("%s", self.last_message)

    def handle_turning(self, _event):
        with self.lock:
            if self.state != "TURNING":
                return
            if self.turn_target_yaw is None or self.turn_started_at is None:
                self.state = "ADVANCING"
                self.schedule_send_goal_locked("ADVANCING", "turn state missing target")
                return

            try:
                _x, _y, current_yaw = self.get_current_xy_yaw()
            except Exception as exc:  # pylint: disable=broad-except
                self.stop_turn_command()
                self.last_message = f"waiting for yaw during turn: {exc}"
                rospy.logwarn_throttle(2.0, "%s", self.last_message)
                return

            error = wrap_angle(self.turn_target_yaw - current_yaw)
            if abs(error) <= self.turn_yaw_tolerance:
                self.stop_turn_command()
                self.turn_target_yaw = None
                self.turn_started_at = None
                self.schedule_send_goal_locked(
                    "ADVANCING",
                    "turn complete, scheduling direction=%s index=%d"
                    % (
                        "forward" if self.route_direction > 0 else "reverse",
                        self.current_index if self.current_index is not None else -1,
                    ),
                )
                return

            elapsed = (rospy.Time.now() - self.turn_started_at).to_sec()
            if self.turn_timeout > 0.0 and elapsed >= self.turn_timeout:
                self.stop_turn_command()
                self.turn_target_yaw = None
                self.turn_started_at = None
                rospy.logwarn(
                    "Turn timeout after %.1fs with yaw_error=%.1fdeg; resuming patrol",
                    elapsed,
                    math.degrees(error),
                )
                self.schedule_send_goal_locked(
                    "ADVANCING",
                    "turn timeout, scheduling direction=%s"
                    % ("forward" if self.route_direction > 0 else "reverse"),
                )
                return

            angular_z = max(
                self.turn_min_angular_speed,
                min(abs(error) * self.turn_kp, self.turn_angular_speed),
            )
            self.publish_turn_command(math.copysign(angular_z, error))

    def maybe_advance_by_pass_radius_locked(self) -> bool:
        if not self.advance_on_pass_radius:
            return False
        if self.current_index is None or self.current_index >= len(self.waypoints):
            return False

        x, y = self.get_current_xy()
        route_progress, nearest_distance = self.closest_route_progress(x, y)
        self.last_route_progress = route_progress
        self.last_nearest_distance = nearest_distance

        if self.try_complete_terminal_segment_locked(route_progress, x, y):
            return True

        checkpoint = self.waypoints[self.current_index]
        checkpoint_distance = math.hypot(checkpoint.x - x, checkpoint.y - y)
        self.last_checkpoint_distance = checkpoint_distance

        terminal_index = self.terminal_index_for_direction()
        is_terminal_checkpoint = self.current_index == terminal_index
        pass_radius = (
            self.final_pass_radius
            if is_terminal_checkpoint and not self.loop
            else self.pass_radius
        )
        if checkpoint_distance > pass_radius:
            return False

        passed_index = self.current_index
        passed_id = checkpoint.waypoint_id
        rospy.loginfo(
            "Passed checkpoint index=%d id=%d distance=%.2fm within radius=%.2fm",
            passed_index,
            passed_id,
            checkpoint_distance,
            pass_radius,
        )

        if is_terminal_checkpoint and not self.loop:
            self.current_index = len(self.waypoints)
            self.active_goal_index = None
            self.active_goal_distance = None
            self.active_goal_x = None
            self.active_goal_y = None
            self.current_retries = 0
            self.state = "FINISHED"
            self.last_message = (
                f"passed final checkpoint index={passed_index},id={passed_id}"
            )
            self.move_base.cancel_all_goals()
            return True

        if self.loop:
            if self.ping_pong and is_terminal_checkpoint:
                self.clear_passed_checkpoint_guard_locked()
                self.start_turnaround_locked(
                    "passed terminal checkpoint index=%d,id=%d" % (passed_index, passed_id)
                )
                return True
            self.remember_passed_checkpoint_locked(passed_index, route_progress)
            if self.ping_pong:
                self.current_index += self.route_direction
            else:
                self.current_index = (self.current_index + 1) % len(self.waypoints)
            self.wrapping_to_start = is_terminal_checkpoint and self.current_index == 0
        else:
            self.remember_passed_checkpoint_locked(passed_index, route_progress)
            self.current_index += self.route_direction
            self.wrapping_to_start = False
        self.current_retries = 0
        self.schedule_send_goal_locked(
            "ADVANCING",
            (
                f"passed checkpoint index={passed_index},id={passed_id},"
                f" scheduling next index={self.current_index}"
            ),
        )
        return True

    def handle_continuous_advance(self, _event):
        with self.lock:
            if self.state != "RUNNING" or self.current_index is None:
                return
            if self.rejoin_mode_active:
                try:
                    route_progress, nearest_distance = self.refresh_route_state_locked()
                except Exception as exc:  # pylint: disable=broad-except
                    rospy.logwarn_throttle(2.0, "Route rejoin update failed: %s", exc)
                    return

                if self.maybe_update_rejoin_mode_locked(route_progress, nearest_distance):
                    self.current_retries = 0
                    try:
                        _x, _y, current_yaw = self.get_current_xy_yaw()
                        self.startup_mode_active, _yaw_error = self.evaluate_startup_mode(
                            route_progress, nearest_distance, current_yaw
                        )
                    except Exception:
                        self.startup_mode_active = False
                    self.startup_mode_start_progress = (
                        route_progress if self.startup_mode_active else None
                    )
                    self.last_message = (
                        f"rejoined route at progress={route_progress:.2f}m "
                        f"route_error={nearest_distance:.2f}m"
                    )
                    rospy.loginfo("%s", self.last_message)
                    self.send_goal_locked()
                    return

                if (rospy.Time.now() - self.last_goal_sent_time).to_sec() < max(
                    self.goal_update_min_interval, self.next_goal_delay
                ):
                    return

                target_distance = self.rejoin_goal_distance(route_progress)
                goal_moved = (
                    self.active_goal_distance is None
                    or abs(target_distance - self.active_goal_distance)
                    >= self.rejoin_goal_update_distance
                )
                if not goal_moved:
                    return

                self.current_retries = 0
                self.last_message = (
                    f"rejoining route progress={route_progress:.2f}m "
                    f"route_error={nearest_distance:.2f}m"
                )
                rospy.loginfo("%s", self.last_message)
                self.send_goal_locked()
                return
            try:
                route_progress, nearest_distance = self.refresh_route_state_locked()
                try:
                    _x, _y, current_yaw = self.get_current_xy_yaw()
                    if self.startup_mode_active:
                        self.maybe_update_startup_mode_locked(route_progress, nearest_distance)
                    else:
                        self.maybe_enter_startup_mode_locked(
                            route_progress, nearest_distance, current_yaw
                        )
                except Exception:
                    pass
                if self.maybe_advance_by_pass_radius_locked():
                    return
            except Exception as exc:  # pylint: disable=broad-except
                rospy.logwarn_throttle(2.0, "Pass-radius checkpoint check failed: %s", exc)
                return
            if self.realign_while_running:
                previous_checkpoint = self.current_index
                try:
                    self.advance_past_reached_waypoints_locked()
                except Exception as exc:  # pylint: disable=broad-except
                    rospy.logwarn_throttle(2.0, "Running checkpoint realign failed: %s", exc)
                    return
                if self.current_index != previous_checkpoint:
                    self.current_retries = 0
                    if self.current_index is None or self.current_index >= len(self.waypoints):
                        self.send_goal_locked()
                        return
                    checkpoint = self.waypoints[self.current_index]
                    self.last_message = (
                        "realigned running checkpoint from index=%d to index=%d,id=%d"
                        % (previous_checkpoint, self.current_index, checkpoint.waypoint_id)
                    )
                    rospy.logwarn("%s", self.last_message)
                    self.send_goal_locked()
                    return
            if not self.continuous_advance:
                return
            if self.current_index >= len(self.waypoints) - 1 and not self.loop:
                try:
                    self.refresh_route_state_locked()
                except Exception:
                    pass
                return
            if (rospy.Time.now() - self.last_goal_sent_time).to_sec() < max(
                self.goal_update_min_interval, self.next_goal_delay
            ):
                return

            previous_checkpoint = self.current_index
            try:
                route_progress, _nearest_distance = self.refresh_route_state_locked()
            except Exception:
                return
            self.advance_past_reached_waypoints_locked()
            if self.last_terminal_mode and self.active_goal_terminal:
                return

            target_distance = self.target_goal_distance(route_progress, self.current_index)
            goal_moved = (
                self.active_goal_distance is None
                or abs(target_distance - self.active_goal_distance) >= self.goal_update_distance
            )
            checkpoint_changed = self.current_index != previous_checkpoint
            if not checkpoint_changed and not goal_moved:
                return

            checkpoint = self.waypoints[self.current_index]
            self.current_retries = 0
            message_prefix = "continuous advance"
            if self.startup_mode_active:
                message_prefix = "startup alignment"
            self.last_message = (
                f"{message_prefix} to checkpoint index={self.current_index},"
                f"id={checkpoint.waypoint_id}"
            )
            rospy.loginfo("%s", self.last_message)
            self.send_goal_locked()

    def handle_goal_done(self, goal_sequence, status, result):  # pylint: disable=unused-argument
        with self.lock:
            if goal_sequence != self.active_goal_sequence:
                rospy.logdebug(
                    "Ignoring stale move_base result for sequence=%s",
                    goal_sequence,
                )
                return
            if self.state not in ("RUNNING", "PAUSED"):
                return
            if self.state == "PAUSED":
                return

            if status == GoalStatus.SUCCEEDED:
                if self.rejoin_mode_active:
                    try:
                        route_progress, nearest_distance = self.refresh_route_state_locked()
                    except Exception as exc:  # pylint: disable=broad-except
                        self.state = "FAILED"
                        self.rejoin_mode_active = False
                        self.last_message = f"rejoin pose refresh failed after success: {exc}"
                        rospy.logerr("%s", self.last_message)
                        return

                    if self.maybe_update_rejoin_mode_locked(route_progress, nearest_distance):
                        self.current_retries = 0
                        try:
                            _x, _y, current_yaw = self.get_current_xy_yaw()
                            self.startup_mode_active, _yaw_error = self.evaluate_startup_mode(
                                route_progress, nearest_distance, current_yaw
                            )
                        except Exception:
                            self.startup_mode_active = False
                        self.startup_mode_start_progress = (
                            route_progress if self.startup_mode_active else None
                        )
                        self.last_message = (
                            f"rejoined route at progress={route_progress:.2f}m "
                            f"route_error={nearest_distance:.2f}m"
                        )
                        rospy.loginfo("%s", self.last_message)
                        self.send_goal_locked()
                        return

                    self.current_retries = 0
                    self.schedule_send_goal_locked(
                        "ADVANCING",
                        (
                            f"rejoin goal reached but route_error={nearest_distance:.2f}m, "
                            "scheduling another route rejoin goal"
                        ),
                    )
                    return

                reached_id = -1
                rospy.loginfo(
                    "Reached patrol active goal distance=%s",
                    "unknown"
                    if self.active_goal_distance is None
                    else f"{self.active_goal_distance:.2f}",
                )
                self.current_retries = 0
                if self.current_index is None:
                    self.state = "FINISHED"
                    self.last_message = "goal succeeded but current index was unknown"
                    return
                try:
                    route_progress, _nearest_distance = self.refresh_route_state_locked()
                    if (
                        self.route_direction > 0
                        and route_progress >= self.waypoints[-1].distance - self.skip_distance
                    ):
                        self.current_index = len(self.waypoints)
                    elif (
                        self.route_direction < 0
                        and route_progress <= self.waypoints[0].distance + self.skip_distance
                    ):
                        self.current_index = -1
                    else:
                        start_index = self.choose_start_index(route_progress)
                        self.current_index = (
                            start_index if start_index is not None else len(self.waypoints)
                        )
                except Exception:
                    self.current_index += self.route_direction
                if self.current_index < 0 or self.current_index >= len(self.waypoints):
                    if self.loop:
                        if self.ping_pong:
                            self.start_turnaround_locked("goal succeeded at route boundary")
                            return
                        self.current_index = 0
                        self.wrapping_to_start = True
                    else:
                        self.state = "FINISHED"
                        self.last_message = "finished route"
                        return
                elif self.current_index != 0:
                    self.wrapping_to_start = False
                self.schedule_send_goal_locked(
                    "ADVANCING",
                    f"reached waypoint id={reached_id}, scheduling next index={self.current_index}",
                )
                return

            status_text = goal_status_name(status)
            if self.current_retries < self.max_retries:
                self.current_retries += 1
                self.last_message = f"retry {self.current_retries}/{self.max_retries} after {status_text}"
                rospy.logwarn("%s", self.last_message)
                self.schedule_send_goal_locked("RETRYING", self.last_message)
                return

            self.state = "FAILED"
            self.rejoin_mode_active = False
            self.startup_mode_active = False
            self.startup_mode_start_progress = None
            self.last_message = f"move_base result {status_text}, retries exhausted"
            rospy.logerr("%s", self.last_message)

    def start_from_current_pose(self) -> TriggerResponse:
        with self.lock:
            if self.state in ("RUNNING", "ADVANCING", "RETRYING"):
                return TriggerResponse(success=False, message="patrol is already running")
            try:
                x, y, current_yaw = self.get_current_xy_yaw()
                progress, nearest_distance = self.closest_route_progress(x, y)
            except Exception as exc:  # pylint: disable=broad-except
                self.state = "FAILED"
                self.last_message = f"failed to locate robot in {self.frame_id}: {exc}"
                return TriggerResponse(success=False, message=self.last_message)

            self.last_route_progress = progress
            self.last_nearest_distance = nearest_distance
            start_index = self.choose_start_index(progress)
            if start_index is None:
                self.state = "FINISHED"
                self.current_index = None
                self.wrapping_to_start = False
                self.last_message = "robot is beyond final waypoint"
                return TriggerResponse(success=True, message=self.last_message)

            self.current_index = start_index
            self.clear_passed_checkpoint_guard_locked()
            self.wrapping_to_start = (
                self.loop
                and not self.ping_pong
                and start_index == 0
                and progress >= self.waypoints[-1].distance - self.skip_distance
            )
            self.rejoin_mode_active = self.evaluate_rejoin_mode(nearest_distance)
            if self.rejoin_mode_active:
                self.startup_mode_active = False
                self.startup_mode_start_progress = None
            else:
                self.startup_mode_active = False
                self.startup_mode_start_progress = None
                self.maybe_enter_startup_mode_locked(
                    progress, nearest_distance, current_yaw
                )
            self.current_retries = 0
            if nearest_distance > self.off_route_warn_distance:
                rospy.logwarn(
                    "Robot is %.2fm from the closest waypoint route projection; starting from index %d anyway",
                    nearest_distance,
                    start_index,
                )
            if self.rejoin_mode_active:
                rospy.loginfo(
                    "Entering route rejoin mode: route_error=%.2fm target_progress=%.2fm lookahead=%.2fm",
                    nearest_distance,
                    progress,
                    self.rejoin_goal_lookahead_distance,
                )
            if self.startup_mode_active:
                yaw_error = self.heading_error_to_route(progress, current_yaw)
                rospy.loginfo(
                    "Entering startup patrol mode: route_error=%.2fm yaw_error=%.1fdeg lookahead=%.2fm",
                    nearest_distance,
                    math.degrees(yaw_error),
                    self.startup_goal_lookahead_distance,
                )
            self.send_goal_locked()
            startup_note = ""
            rejoin_note = ""
            if self.rejoin_mode_active:
                rejoin_note = (
                    f", rejoin_mode=true route_error={nearest_distance:.2f}m"
                    f" target_progress={progress:.2f}m"
                )
            if self.startup_mode_active:
                startup_note = (
                    f", startup_mode=true route_error={nearest_distance:.2f}m"
                    f" yaw_error={math.degrees(yaw_error):.1f}deg"
                )
            return TriggerResponse(
                success=True,
                message=(
                    f"started at index={start_index}, route_progress={progress:.2f}m, "
                    f"route_error={nearest_distance:.2f}m{rejoin_note}{startup_note}"
                ),
            )

    def handle_start(self, _request):
        return self.start_from_current_pose()

    def handle_pause(self, _request):
        with self.lock:
            if self.state not in ("RUNNING", "ADVANCING", "RETRYING", "TURNING"):
                return TriggerResponse(success=False, message=f"cannot pause from state={self.state}")
            self.state = "PAUSED"
            self.last_message = "paused"
            self.move_base.cancel_all_goals()
            self.stop_turn_command()
            return TriggerResponse(success=True, message="paused patrol")

    def handle_resume(self, _request):
        with self.lock:
            if self.state in ("RUNNING", "ADVANCING", "RETRYING", "TURNING"):
                return TriggerResponse(success=False, message="patrol is already running")
            if self.current_index is None or self.state in ("IDLE", "FINISHED", "FAILED"):
                return self.start_from_current_pose()
            if self.state == "PAUSED" and self.turn_target_yaw is not None:
                self.turn_started_at = rospy.Time.now()
                self.state = "TURNING"
                self.last_message = (
                    "resumed turn in place to yaw=%.1fdeg"
                    % math.degrees(self.turn_target_yaw)
                )
                return TriggerResponse(success=True, message=self.last_message)
            self.current_retries = 0
            self.send_goal_locked()
            return TriggerResponse(success=True, message=f"resumed at index={self.current_index}")

    def handle_stop(self, _request):
        with self.lock:
            self.move_base.cancel_all_goals()
            self.stop_turn_command()
            self.state = "STOPPED"
            self.current_index = None
            self.wrapping_to_start = False
            self.rejoin_mode_active = False
            self.startup_mode_active = False
            self.startup_mode_start_progress = None
            self.route_direction = 1
            self.turn_target_yaw = None
            self.turn_started_at = None
            self.active_goal_index = None
            self.active_goal_distance = None
            self.active_goal_x = None
            self.active_goal_y = None
            self.last_checkpoint_distance = None
            self.clear_passed_checkpoint_guard_locked()
            self.current_retries = 0
            self.last_message = "stopped"
            return TriggerResponse(success=True, message="stopped patrol")

    def publish_status(self):
        with self.lock:
            current = "none"
            if self.current_index is not None and self.current_index < len(self.waypoints):
                waypoint = self.waypoints[self.current_index]
                current = (
                    f"index={self.current_index},id={waypoint.waypoint_id},"
                    f"x={waypoint.x:.2f},y={waypoint.y:.2f}"
                )
            active_goal = "none"
            if self.active_goal_index is not None and self.active_goal_index < len(self.waypoints):
                waypoint = self.waypoints[self.active_goal_index]
                active_goal = (
                    f"index={self.active_goal_index},id={waypoint.waypoint_id},"
                    f"x={waypoint.x:.2f},y={waypoint.y:.2f}"
                )
            elif self.active_goal_distance is not None:
                active_goal = (
                    f"distance={self.active_goal_distance:.2f},"
                    f"x={self.active_goal_x:.2f},y={self.active_goal_y:.2f}"
                )
            nearest = (
                "unknown"
                if self.last_nearest_distance is None
                else f"{self.last_nearest_distance:.2f}"
            )
            progress = (
                "unknown"
                if self.last_route_progress is None
                else f"{self.last_route_progress:.2f}"
            )
            checkpoint_distance = (
                "unknown"
                if self.last_checkpoint_distance is None
                else f"{self.last_checkpoint_distance:.2f}"
            )
            msg = String()
            msg.data = (
                f"state={self.state} checkpoint={current} active_goal={active_goal} "
                f"total={len(self.waypoints)} "
                f"loop={str(self.loop).lower()} "
                f"ping_pong={str(self.ping_pong).lower()} "
                f"direction={'forward' if self.route_direction > 0 else 'reverse'} "
                f"rejoin_mode={str(self.rejoin_mode_active).lower()} "
                f"startup_mode={str(self.startup_mode_active).lower()} "
                f"terminal_mode={str(self.last_terminal_mode).lower()} "
                f"wrapping_to_start={str(self.wrapping_to_start).lower()} "
                f"progress={progress} route_error={nearest} "
                f"checkpoint_distance={checkpoint_distance} "
                f"message={self.last_message}"
            )
            self.status_pub.publish(msg)


def main() -> int:
    rospy.init_node("waypoint_patrol")
    try:
        WaypointPatrolNode()
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logerr("Failed to start waypoint patrol: %s", exc)
        return 1
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())

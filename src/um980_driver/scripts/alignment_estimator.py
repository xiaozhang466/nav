#!/usr/bin/env python3
"""Online alignment estimator for map <-> rtk_map.

Replaces the static ``static_transform_from_params.py`` node for the
map_rtk_alignment use-case, adding automatic re-calibration:

Startup
  1. Load cached alignment from YAML (if available).
  2. If ``publish_cached_immediately`` is True, publish cached TF right away
     (needed when rtk_lidar_relocalizer is enabled).
     If False (default), withhold TF until verification — RTK is blocked.
  3. Wait ``stabilization_delay`` seconds for LiDAR localization to converge.
  4. Collect matched RTK / LiDAR position pairs and verify cached alignment.
     - If cache is valid (small residuals): publish TF, done.
     - If cache is stale:  clear samples, re-estimate from scratch,
       publish new TF once ready.
  5. Continue monitoring; re-estimate on drift.
"""

from __future__ import annotations

import datetime
import math
import os
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# ── helpers ──────────────────────────────────────────────────────────

@dataclass
class Sample:
    stamp: float
    x: float
    y: float


def quaternion_to_yaw(o) -> float:
    return math.atan2(
        2.0 * (o.w * o.z + o.x * o.y),
        1.0 - 2.0 * (o.y * o.y + o.z * o.z),
    )


def quaternion_from_yaw(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def fit_se2(ref, tgt):
    """Fit T such that  tgt ≈ T · ref.  Returns (yaw, tx, ty)."""
    n = len(ref)
    if n < 3:
        return None
    rcx = statistics.fmean(p[0] for p in ref)
    rcy = statistics.fmean(p[1] for p in ref)
    tcx = statistics.fmean(p[0] for p in tgt)
    tcy = statistics.fmean(p[1] for p in tgt)
    cross = dot = 0.0
    for (rx, ry), (tx, ty) in zip(ref, tgt):
        dx_r, dy_r = rx - rcx, ry - rcy
        dx_t, dy_t = tx - tcx, ty - tcy
        dot += dx_r * dx_t + dy_r * dy_t
        cross += dx_r * dy_t - dy_r * dx_t
    theta = math.atan2(cross, dot)
    c, s = math.cos(theta), math.sin(theta)
    tx = tcx - (c * rcx - s * rcy)
    ty = tcy - (s * rcx + c * rcy)
    return theta, tx, ty


def apply_se2(yaw, tx, ty, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y + tx, s * x + c * y + ty


def compute_residuals(yaw, tx, ty, ref_pts, tgt_pts):
    residuals = []
    for (rx, ry), (lx, ly) in zip(ref_pts, tgt_pts):
        ax, ay = apply_se2(yaw, tx, ty, rx, ry)
        residuals.append(math.hypot(ax - lx, ay - ly))
    return residuals


def rmse(residuals):
    if not residuals:
        return float("inf")
    return math.sqrt(statistics.fmean(r * r for r in residuals))


# ── YAML I/O ─────────────────────────────────────────────────────────

def load_alignment(path: str):
    """Returns (tx, ty, yaw) or None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        t = data.get("translation", {})
        r = data.get("rotation", {})
        tx = float(t.get("x", 0.0))
        ty = float(t.get("y", 0.0))
        yaw = float(r.get("yaw", 0.0))
        return tx, ty, yaw
    except Exception as exc:
        rospy.logwarn("Failed to load alignment from %s: %s", path, exc)
        return None


def save_alignment(path: str, tx: float, ty: float, yaw: float,
                   n_samples: int, travel: float, rmse_val: float):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    content = (
        "# Static calibration from the LiDAR map frame into the fixed RTK datum frame.\n"
        "#\n"
        f"# Auto-estimated on {now}\n"
        f"# Samples: {n_samples}  Travel: {travel:.1f}m  RMSE: {rmse_val:.4f}m\n"
        "#\n"
        "\n"
        "parent_frame: map\n"
        "child_frame: rtk_map\n"
        "\n"
        "translation:\n"
        f"  x: {tx:.6f}\n"
        f"  y: {ty:.6f}\n"
        "  z: 0.0\n"
        "\n"
        "rotation:\n"
        "  roll: 0.0\n"
        "  pitch: 0.0\n"
        f"  yaw: {yaw:.6f}\n"
    )
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


# ── main node ────────────────────────────────────────────────────────

class AlignmentEstimator:

    # ── lifecycle ────────────────────────────────────────────────────

    def __init__(self):
        # parameters
        self.alignment_file = rospy.get_param(
            "~alignment_file",
            os.path.join(
                os.path.dirname(__file__),
                "..", "config", "map_rtk_alignment.yaml",
            ),
        )
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "rtk_map")
        self.rtk_odom_topic = rospy.get_param("~rtk_odom_topic", "/odometry/rtk_map")
        self.lidar_odom_topic = rospy.get_param("~lidar_odom_topic", "/odometry/lidar_map")
        self.rtk_fix_type_topic = rospy.get_param("~rtk_fix_type_topic", "/rtk/fix_type")
        self.lidar_status_topic = rospy.get_param(
            "~lidar_status_topic", "/lidar_localization/status"
        )
        self.publish_cached_immediately = bool(
            rospy.get_param("~publish_cached_immediately", False)
        )

        self.min_samples = int(rospy.get_param("~min_samples", 30))
        self.min_travel_m = float(rospy.get_param("~min_travel_m", 15.0))
        self.sample_spacing_m = float(rospy.get_param("~sample_spacing_m", 0.5))
        self.max_time_delta = float(rospy.get_param("~max_time_delta", 0.15))
        self.max_accept_rmse = float(rospy.get_param("~max_accept_rmse", 0.15))
        self.max_auto_update_delta = float(rospy.get_param("~max_auto_update_delta", 1.0))

        # verification: how many samples needed to verify cached alignment
        self.verify_min_samples = int(rospy.get_param("~verify_min_samples", 8))
        self.verify_min_travel = float(rospy.get_param("~verify_min_travel", 3.0))
        self.verify_max_residual = float(rospy.get_param("~verify_max_residual", 0.3))

        # stabilization: wait for LiDAR localization to converge before
        # collecting samples.  Starts counting from first lidar_ok=True.
        self.stabilization_delay = float(rospy.get_param("~stabilization_delay", 15.0))

        # re-estimation interval
        self.update_interval = float(rospy.get_param("~update_interval", 300.0))
        self.sliding_window_size = int(rospy.get_param("~sliding_window_size", 200))

        # how many consecutive high-RMSE attempts before clearing window
        self.max_bad_estimates = int(rospy.get_param("~max_bad_estimates", 10))

        self.status_interval = float(rospy.get_param("~status_interval", 2.0))

        # state
        self.lock = threading.Lock()
        self.cached_alignment: Optional[tuple] = None  # (tx, ty, yaw)
        self.active_alignment: Optional[tuple] = None   # currently published
        self.tf_published = False
        self.verified = False

        self.rtk_history: deque[Sample] = deque(maxlen=5000)
        self.matched_rtk: deque[tuple] = deque(maxlen=self.sliding_window_size)
        self.matched_lidar: deque[tuple] = deque(maxlen=self.sliding_window_size)
        self.last_kept: Optional[tuple] = None
        self.travel = 0.0
        self.total_matched = 0
        self.bad_estimate_count = 0

        self.rtk_fix_ok = False
        self.lidar_ok = False
        self.lidar_ok_since: Optional[float] = None  # wall time when lidar_ok first became True
        self.stabilized = False
        self.last_estimate_time = 0.0
        self.last_status_time = 0.0

        # TF
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # pub / sub
        self.status_pub = rospy.Publisher(
            "~status", String, queue_size=5, latch=True
        )
        rospy.Subscriber(
            self.rtk_odom_topic, Odometry, self._rtk_cb, queue_size=100
        )
        rospy.Subscriber(
            self.lidar_odom_topic, Odometry, self._lidar_cb, queue_size=100
        )
        rospy.Subscriber(
            self.rtk_fix_type_topic, String, self._fix_type_cb, queue_size=10
        )
        rospy.Subscriber(
            self.lidar_status_topic, String, self._lidar_status_cb, queue_size=10
        )

        # load cache
        self.cached_alignment = load_alignment(self.alignment_file)
        if self.cached_alignment:
            tx, ty, yaw = self.cached_alignment
            if self.publish_cached_immediately:
                rospy.loginfo(
                    "Loaded cached alignment: tx=%.3f ty=%.3f yaw=%.3f°. "
                    "Publishing TF immediately (verification pending).",
                    tx, ty, math.degrees(yaw),
                )
                self._publish_tf(tx, ty, yaw)
                self.active_alignment = self.cached_alignment
            else:
                rospy.loginfo(
                    "Loaded cached alignment: tx=%.3f ty=%.3f yaw=%.3f°. "
                    "Withholding TF until verified (RTK blocked).",
                    tx, ty, math.degrees(yaw),
                )
        else:
            rospy.logwarn(
                "No cached alignment found at %s — "
                "will estimate from scratch.",
                self.alignment_file,
            )

        self._publish_status("startup")

        # periodic check
        rospy.Timer(rospy.Duration(self.status_interval), self._timer_cb)

    # ── callbacks ────────────────────────────────────────────────────

    def _fix_type_cb(self, msg: String):
        self.rtk_fix_ok = msg.data.strip().lower() == "rtk_fixed"

    def _lidar_status_cb(self, msg: String):
        parts = {}
        for p in msg.data.split():
            if "=" in p:
                k, v = p.split("=", 1)
                parts[k] = v
        was_ok = self.lidar_ok
        self.lidar_ok = parts.get("ok", "false").lower() == "true"

        # Track when lidar first became OK for stabilization delay
        if self.lidar_ok and not was_ok:
            self.lidar_ok_since = rospy.get_time()
            self.stabilized = False
            rospy.loginfo("LiDAR localization became OK, starting stabilization timer (%.1fs)",
                          self.stabilization_delay)
        elif not self.lidar_ok:
            self.lidar_ok_since = None
            self.stabilized = False

    def _rtk_cb(self, msg: Odometry):
        s = Sample(
            msg.header.stamp.to_sec(),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        with self.lock:
            self.rtk_history.append(s)

    def _lidar_cb(self, msg: Odometry):
        s = Sample(
            msg.header.stamp.to_sec(),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        with self.lock:
            self._try_match(s)

    # ── matching ─────────────────────────────────────────────────────

    def _interp_rtk(self, stamp: float) -> Optional[Sample]:
        hist = list(self.rtk_history)
        if len(hist) < 2:
            return None
        if stamp < hist[0].stamp - self.max_time_delta:
            return None
        if stamp > hist[-1].stamp + self.max_time_delta:
            return None
        for i in range(1, len(hist)):
            if hist[i].stamp >= stamp:
                dt = hist[i].stamp - hist[i - 1].stamp
                if dt <= 0 or dt > self.max_time_delta * 3:
                    return None
                r = (stamp - hist[i - 1].stamp) / dt
                return Sample(
                    stamp,
                    hist[i - 1].x + r * (hist[i].x - hist[i - 1].x),
                    hist[i - 1].y + r * (hist[i].y - hist[i - 1].y),
                )
        return None

    def _check_stabilization(self) -> bool:
        """Return True when LiDAR has been OK long enough to trust."""
        if self.stabilized:
            return True
        if not self.lidar_ok or self.lidar_ok_since is None:
            return False
        elapsed = rospy.get_time() - self.lidar_ok_since
        if elapsed >= self.stabilization_delay:
            self.stabilized = True
            rospy.loginfo("LiDAR stabilized after %.1fs. Starting sample collection.", elapsed)
            return True
        return False

    def _try_match(self, lidar: Sample):
        if not self.rtk_fix_ok:
            return
        if not self.lidar_ok:
            return
        # Wait for LiDAR to stabilize before collecting samples
        if not self._check_stabilization():
            return

        rtk = self._interp_rtk(lidar.stamp)
        if rtk is None:
            return

        pt = (rtk.x, rtk.y)
        if self.last_kept is not None:
            d = math.hypot(pt[0] - self.last_kept[0], pt[1] - self.last_kept[1])
            if d < self.sample_spacing_m:
                return
            self.travel += d
        self.matched_rtk.append(pt)
        self.matched_lidar.append((lidar.x, lidar.y))
        self.last_kept = pt
        self.total_matched += 1

        # try verification / estimation after each new match
        self._check_state()

    # ── state machine ────────────────────────────────────────────────

    def _check_state(self):
        n = len(self.matched_rtk)

        # Phase 1: verify cached alignment
        if not self.verified and self.cached_alignment is not None:
            if n >= self.verify_min_samples and self.travel >= self.verify_min_travel:
                self._try_verify_cache()
            return

        # Phase 2: estimate from scratch (no cache or cache failed)
        if not self.verified:
            if n >= self.min_samples and self.travel >= self.min_travel_m:
                self._estimate_and_publish()
            return

        # Phase 3: continuous monitoring / re-estimation
        now = rospy.get_time()
        if now - self.last_estimate_time >= self.update_interval:
            if n >= self.min_samples and self.travel >= self.min_travel_m:
                self._re_estimate()

    def _try_verify_cache(self):
        tx, ty, yaw = self.cached_alignment
        rtk_pts = list(self.matched_rtk)
        lidar_pts = list(self.matched_lidar)

        residuals = compute_residuals(yaw, tx, ty, rtk_pts, lidar_pts)
        r = rmse(residuals)
        max_r = max(residuals)

        if r <= self.verify_max_residual:
            rospy.loginfo(
                "Cached alignment VERIFIED (rmse=%.4fm max=%.4fm, %d samples, %.1fm).",
                r, max_r, len(rtk_pts), self.travel,
            )
            self.active_alignment = self.cached_alignment
            self.verified = True
            if not self.tf_published:
                self._publish_tf(tx, ty, yaw)
            self.last_estimate_time = rospy.get_time()
            self._publish_status("verified_cache")
            # Reset window for future monitoring
            self._reset_window()
        else:
            rospy.logwarn(
                "Cached alignment REJECTED (rmse=%.4fm max=%.4fm > threshold %.3fm). "
                "Will re-estimate from scratch.",
                r, max_r, self.verify_max_residual,
            )
            self.cached_alignment = None
            # Clear contaminated samples so Phase 2 starts fresh
            self._reset_window()
            self._publish_status("cache_rejected")

    def _estimate_and_publish(self):
        rtk_pts = list(self.matched_rtk)
        lidar_pts = list(self.matched_lidar)
        result = fit_se2(rtk_pts, lidar_pts)
        if result is None:
            return

        yaw, tx, ty = result
        residuals = compute_residuals(yaw, tx, ty, rtk_pts, lidar_pts)
        r = rmse(residuals)

        if r > self.max_accept_rmse:
            self.bad_estimate_count += 1
            rospy.logwarn(
                "Estimated alignment has high RMSE %.4fm (threshold %.3fm), "
                "attempt %d/%d.",
                r, self.max_accept_rmse,
                self.bad_estimate_count, self.max_bad_estimates,
            )
            if self.bad_estimate_count >= self.max_bad_estimates:
                rospy.logwarn(
                    "Too many bad estimates (%d). Clearing window and retrying.",
                    self.bad_estimate_count,
                )
                self._reset_window()
                self.bad_estimate_count = 0
            return

        self.bad_estimate_count = 0
        rospy.loginfo(
            "New alignment estimated: tx=%.3f ty=%.3f yaw=%.3f° "
            "rmse=%.4fm (%d samples, %.1fm). Publishing TF.",
            tx, ty, math.degrees(yaw), r, len(rtk_pts), self.travel,
        )
        self.active_alignment = (tx, ty, yaw)
        self.verified = True
        self._publish_tf(tx, ty, yaw)
        self.last_estimate_time = rospy.get_time()
        save_alignment(self.alignment_file, tx, ty, yaw,
                       len(rtk_pts), self.travel, r)
        rospy.loginfo("Alignment saved to %s", self.alignment_file)
        self._publish_status("estimated_new")
        # reset sliding window for future monitoring
        self._reset_window()

    def _re_estimate(self):
        rtk_pts = list(self.matched_rtk)
        lidar_pts = list(self.matched_lidar)
        if len(rtk_pts) < self.min_samples:
            return
        result = fit_se2(rtk_pts, lidar_pts)
        if result is None:
            return

        yaw, tx, ty = result
        residuals = compute_residuals(yaw, tx, ty, rtk_pts, lidar_pts)
        r = rmse(residuals)

        self.last_estimate_time = rospy.get_time()

        if r > self.max_accept_rmse:
            rospy.logwarn(
                "Re-estimation RMSE too high: %.4fm. Keeping current alignment.",
                r,
            )
            self._reset_window()
            return

        if self.active_alignment is not None:
            old_tx, old_ty, old_yaw = self.active_alignment
            delta = math.hypot(tx - old_tx, ty - old_ty)
            if delta > self.max_auto_update_delta:
                rospy.logwarn(
                    "Re-estimated alignment differs by %.3fm from current "
                    "(threshold %.3fm). NOT auto-updating. "
                    "New values: tx=%.3f ty=%.3f yaw=%.3f°",
                    delta, self.max_auto_update_delta,
                    tx, ty, math.degrees(yaw),
                )
                self._publish_status("drift_warning")
                self._reset_window()
                return

        rospy.loginfo(
            "Alignment updated: tx=%.3f ty=%.3f yaw=%.3f° rmse=%.4fm "
            "(%d samples, %.1fm)",
            tx, ty, math.degrees(yaw), r, len(rtk_pts), self.travel,
        )
        self.active_alignment = (tx, ty, yaw)
        self._publish_tf(tx, ty, yaw)
        save_alignment(self.alignment_file, tx, ty, yaw,
                       len(rtk_pts), self.travel, r)
        self._publish_status("updated")
        self._reset_window()

    def _reset_window(self):
        self.matched_rtk.clear()
        self.matched_lidar.clear()
        self.last_kept = None
        self.travel = 0.0

    # ── TF publishing ────────────────────────────────────────────────

    def _publish_tf(self, tx: float, ty: float, yaw: float):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)
        self.tf_published = True
        rospy.loginfo(
            "Published static TF %s -> %s: tx=%.4f ty=%.4f yaw=%.4f",
            self.parent_frame, self.child_frame, tx, ty, yaw,
        )

    # ── status ───────────────────────────────────────────────────────

    def _publish_status(self, reason: str):
        status = "verified=%s tf_published=%s samples=%d travel=%.1fm reason=%s" % (
            str(self.verified).lower(),
            str(self.tf_published).lower(),
            self.total_matched,
            self.travel,
            reason,
        )
        self.status_pub.publish(String(data=status))

    def _timer_cb(self, _event):
        now = rospy.get_time()
        if now - self.last_status_time < self.status_interval:
            return
        self.last_status_time = now

        with self.lock:
            n = len(self.matched_rtk)

        if not self.verified:
            if self.cached_alignment is not None:
                state = "verifying_cache"
            elif not self.stabilized:
                state = "waiting_for_stabilization"
            else:
                state = "estimating"
            rospy.loginfo(
                "Alignment: %s | samples=%d travel=%.1fm "
                "rtk_ok=%s lidar_ok=%s stabilized=%s",
                state, n, self.travel,
                self.rtk_fix_ok, self.lidar_ok, self.stabilized,
            )
        self._publish_status("monitoring")


def main():
    rospy.init_node("alignment_estimator", anonymous=False)
    AlignmentEstimator()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

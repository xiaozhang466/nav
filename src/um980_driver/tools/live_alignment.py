#!/usr/bin/env python3
"""Live estimation of the map <-> rtk_map alignment transform.

Subscribe to RTK and LiDAR odometry simultaneously, collect matched position
pairs while the robot drives, and fit a 2D rigid transform (SE2).

Usage:
    rosrun um980_driver live_alignment.py

Drive the robot around for 20-30m. The script prints progress and outputs the
new alignment when enough samples are collected. Press Ctrl+C at any time to
get the current best estimate.
"""

from __future__ import annotations

import math
import signal
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import String


@dataclass
class Sample:
    stamp: float
    x: float
    y: float
    yaw: float


def quaternion_to_yaw(o) -> float:
    return math.atan2(
        2.0 * (o.w * o.z + o.x * o.y),
        1.0 - 2.0 * (o.y * o.y + o.z * o.z),
    )


def wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def fit_se2(ref, tgt):
    """Fit T such that tgt ~= T * ref.  Returns (yaw, tx, ty)."""
    n = len(ref)
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


class LiveAligner:
    def __init__(self):
        self.rtk_odom_topic = rospy.get_param("~rtk_odom_topic", "/odometry/rtk_map")
        self.lidar_odom_topic = rospy.get_param("~lidar_odom_topic", "/odometry/lidar_map")
        self.rtk_fix_type_topic = rospy.get_param("~rtk_fix_type_topic", "/rtk/fix_type")
        self.min_samples = int(rospy.get_param("~min_samples", 20))
        self.min_travel = float(rospy.get_param("~min_travel", 10.0))
        self.sample_spacing = float(rospy.get_param("~sample_spacing", 0.5))
        self.max_time_delta = float(rospy.get_param("~max_time_delta", 0.15))
        self.output_file = rospy.get_param(
            "~output_file",
            rospy.get_param(
                "~config",
                "/home/sigu/RTK/421-1/src/um980_driver/config/map_rtk_alignment.yaml",
            ),
        )

        self.rtk_history: deque[Sample] = deque(maxlen=5000)
        self.lidar_history: deque[Sample] = deque(maxlen=5000)
        self.matched_rtk: list[tuple[float, float]] = []
        self.matched_lidar: list[tuple[float, float]] = []
        self.last_kept: tuple[float, float] | None = None
        self.travel = 0.0
        self.rtk_fix_ok = False
        self.lock = threading.Lock()

        rospy.Subscriber(self.rtk_odom_topic, Odometry, self.rtk_cb, queue_size=100)
        rospy.Subscriber(self.lidar_odom_topic, Odometry, self.lidar_cb, queue_size=100)
        rospy.Subscriber(self.rtk_fix_type_topic, String, self.fix_type_cb, queue_size=10)

        rospy.loginfo("Live aligner started. Drive the robot around (>%.0fm)...", self.min_travel)

    def fix_type_cb(self, msg: String):
        self.rtk_fix_ok = msg.data.strip().lower() == "rtk_fixed"

    def rtk_cb(self, msg: Odometry):
        s = Sample(
            msg.header.stamp.to_sec(),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )
        with self.lock:
            self.rtk_history.append(s)

    def lidar_cb(self, msg: Odometry):
        s = Sample(
            msg.header.stamp.to_sec(),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quaternion_to_yaw(msg.pose.pose.orientation),
        )
        with self.lock:
            self.lidar_history.append(s)
            self._try_match(s)

    def _interp_rtk(self, stamp: float) -> Sample | None:
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
                    0,
                )
        return None

    def _try_match(self, lidar: Sample):
        if not self.rtk_fix_ok:
            return
        rtk = self._interp_rtk(lidar.stamp)
        if rtk is None:
            return
        pt = (rtk.x, rtk.y)
        if self.last_kept is not None:
            d = math.hypot(pt[0] - self.last_kept[0], pt[1] - self.last_kept[1])
            if d < self.sample_spacing:
                return
            self.travel += d
        self.matched_rtk.append(pt)
        self.matched_lidar.append((lidar.x, lidar.y))
        self.last_kept = pt

    def result(self):
        with self.lock:
            n = len(self.matched_rtk)
            if n < 3:
                return None, n, self.travel
            yaw, tx, ty = fit_se2(self.matched_rtk, self.matched_lidar)
            aligned = [apply_se2(yaw, tx, ty, x, y) for x, y in self.matched_rtk]
            residuals = [
                math.hypot(ax - lx, ay - ly)
                for (ax, ay), (lx, ly) in zip(aligned, self.matched_lidar)
            ]
            rmse = math.sqrt(statistics.fmean(r * r for r in residuals))
            return {
                "yaw": yaw,
                "tx": tx,
                "ty": ty,
                "rmse": rmse,
                "max_res": max(residuals),
                "n": n,
                "travel": self.travel,
            }, n, self.travel

    def save(self, r):
        import datetime
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        content = (
            f"# Static calibration from the LiDAR map frame into the fixed RTK datum frame.\n"
            f"#\n"
            f"# Live-estimated on {now}\n"
            f"# Samples: {r['n']}  Travel: {r['travel']:.1f}m  RMSE: {r['rmse']:.4f}m\n"
            f"#\n"
            f"\n"
            f"parent_frame: map\n"
            f"child_frame: rtk_map\n"
            f"\n"
            f"translation:\n"
            f"  x: {r['tx']:.6f}\n"
            f"  y: {r['ty']:.6f}\n"
            f"  z: 0.0\n"
            f"\n"
            f"rotation:\n"
            f"  roll: 0.0\n"
            f"  pitch: 0.0\n"
            f"  yaw: {r['yaw']:.6f}\n"
        )
        with open(self.output_file, "w") as f:
            f.write(content)
        rospy.loginfo("Saved alignment to %s", self.output_file)


def main():
    rospy.init_node("live_alignment", anonymous=False)
    aligner = LiveAligner()

    rate = rospy.Rate(1.0)
    done = False

    def on_shutdown(*_):
        nonlocal done
        done = True

    signal.signal(signal.SIGINT, on_shutdown)
    rospy.on_shutdown(on_shutdown)

    while not done and not rospy.is_shutdown():
        r, n, travel = aligner.result()
        if r is None:
            rospy.loginfo("Collecting... samples=%d travel=%.1fm (need %d samples, %.0fm)",
                          n, travel, aligner.min_samples, aligner.min_travel)
        else:
            rospy.loginfo(
                "samples=%d travel=%.1fm | yaw=%.4f° tx=%.3f ty=%.3f RMSE=%.4fm max=%.4fm",
                r["n"], r["travel"],
                math.degrees(r["yaw"]), r["tx"], r["ty"],
                r["rmse"], r["max_res"],
            )
            if r["n"] >= aligner.min_samples and r["travel"] >= aligner.min_travel:
                rospy.loginfo("=== READY! Enough data collected. ===")
                rospy.loginfo("Press Ctrl+C to accept and save, or keep driving for better accuracy.")
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break

    r, n, travel = aligner.result()
    if r is not None and n >= 3:
        print("\n" + "=" * 60)
        print("Final alignment result")
        print("=" * 60)
        print(f"  Samples:  {r['n']}")
        print(f"  Travel:   {r['travel']:.1f}m")
        print(f"  tx:       {r['tx']:.6f}")
        print(f"  ty:       {r['ty']:.6f}")
        print(f"  yaw:      {r['yaw']:.6f} rad  ({math.degrees(r['yaw']):.3f} deg)")
        print(f"  RMSE:     {r['rmse']:.4f}m")
        print(f"  Max res:  {r['max_res']:.4f}m")
        print("=" * 60)

        try:
            ans = input("\nSave to %s? [Y/n] " % aligner.output_file).strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("", "y", "yes"):
            aligner.save(r)
            print("Done! Restart the system to use the new alignment.")
        else:
            print("Not saved.")
    else:
        print("\nNot enough data to estimate alignment (need at least 3 matched samples).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

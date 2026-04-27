#!/usr/bin/env python3
"""Estimate planar wheel-odometry covariance from a rosbag.

This tool uses:
  - wheel odometry pose/twist from nav_msgs/Odometry
  - single-antenna RTK positions from sensor_msgs/NavSatFix
  - IMU yaw-rate from sensor_msgs/Imu
  - the recorded base_link -> gps_link static transform

The estimates are pragmatic, not ground-truth calibration:
  - x/y use RTK position after a 2D rigid alignment
  - vx uses RTK position derivatives projected into the odom body frame
  - vyaw uses IMU angular_velocity.z as a short-term reference
  - yaw uses RTK course-over-ground during moving segments only
  - vy is copied from vx unless explicitly requested otherwise
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
import statistics
import sys
from typing import Iterable

import rosbag


EARTH_RADIUS_M = 6378137.0


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def variance(values: Iterable[float]) -> float | None:
    sequence = list(values)
    if not sequence:
        return None
    if len(sequence) == 1:
        return 0.0
    return statistics.pvariance(sequence)


def mean(values: Iterable[float]) -> float | None:
    sequence = list(values)
    if not sequence:
        return None
    return statistics.fmean(sequence)


def stddev(values: Iterable[float]) -> float | None:
    sequence = list(values)
    if not sequence:
        return None
    if len(sequence) == 1:
        return 0.0
    return statistics.pstdev(sequence)


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def latlon_to_local_xy(
    lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float
) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = (lon - lon0) * math.cos(lat0) * EARTH_RADIUS_M
    y = (lat - lat0) * EARTH_RADIUS_M
    return x, y


def interpolate_scalar(times: list[float], values: list[float], stamp: float) -> float | None:
    if not times or stamp < times[0] or stamp > times[-1]:
        return None
    idx = bisect.bisect_left(times, stamp)
    if idx == 0:
        return values[0]
    if idx == len(times):
        return values[-1]
    if times[idx] == stamp:
        return values[idx]
    t0 = times[idx - 1]
    t1 = times[idx]
    if t1 <= t0:
        return values[idx - 1]
    ratio = (stamp - t0) / (t1 - t0)
    return values[idx - 1] + ratio * (values[idx] - values[idx - 1])


def interpolate_angle(times: list[float], values: list[float], stamp: float) -> float | None:
    if not times or stamp < times[0] or stamp > times[-1]:
        return None
    idx = bisect.bisect_left(times, stamp)
    if idx == 0:
        return values[0]
    if idx == len(times):
        return values[-1]
    if times[idx] == stamp:
        return values[idx]
    t0 = times[idx - 1]
    t1 = times[idx]
    if t1 <= t0:
        return values[idx - 1]
    ratio = (stamp - t0) / (t1 - t0)
    delta = wrap_angle(values[idx] - values[idx - 1])
    return wrap_angle(values[idx - 1] + ratio * delta)


def compute_rtk_derivatives(
    times: list[float], xs: list[float], ys: list[float]
) -> tuple[list[float | None], list[float | None], list[float | None], list[float | None]]:
    vx: list[float | None] = [None] * len(times)
    vy: list[float | None] = [None] * len(times)
    speed: list[float | None] = [None] * len(times)
    course: list[float | None] = [None] * len(times)
    if len(times) < 3:
        return vx, vy, speed, course

    for i in range(1, len(times) - 1):
        dt = times[i + 1] - times[i - 1]
        if dt <= 0.0:
            continue
        vx_i = (xs[i + 1] - xs[i - 1]) / dt
        vy_i = (ys[i + 1] - ys[i - 1]) / dt
        vx[i] = vx_i
        vy[i] = vy_i
        speed_i = math.hypot(vx_i, vy_i)
        speed[i] = speed_i
        if speed_i > 1e-9:
            course[i] = math.atan2(vy_i, vx_i)
    return vx, vy, speed, course


def fit_se2(
    ref_points: list[tuple[float, float]], odom_points: list[tuple[float, float]]
) -> tuple[float, float, float]:
    if len(ref_points) != len(odom_points) or len(ref_points) < 2:
        raise ValueError("need at least two matched points for SE(2) fit")

    ref_cx = statistics.fmean(point[0] for point in ref_points)
    ref_cy = statistics.fmean(point[1] for point in ref_points)
    odom_cx = statistics.fmean(point[0] for point in odom_points)
    odom_cy = statistics.fmean(point[1] for point in odom_points)

    cross = 0.0
    dot = 0.0
    for (rx, ry), (ox, oy) in zip(ref_points, odom_points):
        rcx = rx - ref_cx
        rcy = ry - ref_cy
        ocx = ox - odom_cx
        ocy = oy - odom_cy
        dot += rcx * ocx + rcy * ocy
        cross += rcx * ocy - rcy * ocx

    theta = math.atan2(cross, dot)
    c = math.cos(theta)
    s = math.sin(theta)
    tx = odom_cx - (c * ref_cx - s * ref_cy)
    ty = odom_cy - (s * ref_cx + c * ref_cy)
    return theta, tx, ty


def apply_se2(theta: float, tx: float, ty: float, x: float, y: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x - s * y + tx, s * x + c * y + ty


def circular_mean(angles: Iterable[float]) -> float | None:
    sequence = list(angles)
    if not sequence:
        return None
    sin_mean = statistics.fmean(math.sin(angle) for angle in sequence)
    cos_mean = statistics.fmean(math.cos(angle) for angle in sequence)
    return math.atan2(sin_mean, cos_mean)


def integrate_imu_yaw(times: list[float], wz_values: list[float]) -> list[float]:
    if not times:
        return []
    yaws = [0.0]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0.0:
            yaws.append(yaws[-1])
            continue
        yaw = yaws[-1] + 0.5 * (wz_values[i - 1] + wz_values[i]) * dt
        yaws.append(wrap_angle(yaw))
    return yaws


def extract_base_to_gps_tf(
    bag: rosbag.Bag, base_frame: str, gps_frame: str
) -> tuple[float, float] | None:
    for _, msg, _ in bag.read_messages(topics=["/tf_static"]):
        for transform in msg.transforms:
            if (
                transform.header.frame_id == base_frame
                and transform.child_frame_id == gps_frame
            ):
                return transform.transform.translation.x, transform.transform.translation.y
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate planar wheel-odometry covariance from wheel odom, RTK, and IMU."
    )
    parser.add_argument("bag", help="Path to the rosbag file")
    parser.add_argument("--odom-topic", default="/ranger_base_node/odom")
    parser.add_argument("--imu-topic", default="/IMU_data")
    parser.add_argument("--fix-topic", default="/rtk/fix")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--gps-frame", default="gps_link")
    parser.add_argument("--copy-vy-from-vx", action="store_true", default=True)
    parser.add_argument("--min-rtk-speed-for-course", type=float, default=0.15)
    parser.add_argument("--min-rtk-speed-for-vx", type=float, default=0.10)
    parser.add_argument("--min-turn-rate-for-vyaw", type=float, default=0.02)
    parser.add_argument("--stationary-linear-threshold", type=float, default=0.03)
    parser.add_argument("--stationary-angular-threshold", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        print(f"bag file not found: {bag_path}", file=sys.stderr)
        return 1

    odom_times: list[float] = []
    odom_x: list[float] = []
    odom_y: list[float] = []
    odom_yaw: list[float] = []
    odom_vx: list[float] = []
    odom_vy: list[float] = []
    odom_wz: list[float] = []

    imu_times: list[float] = []
    imu_wz: list[float] = []

    fix_times: list[float] = []
    fix_lat: list[float] = []
    fix_lon: list[float] = []
    fix_status: list[int] = []

    with rosbag.Bag(bag_path, "r") as bag:
        gps_offset = extract_base_to_gps_tf(bag, args.base_frame, args.gps_frame)
        for topic, msg, t in bag.read_messages(topics=[args.odom_topic, args.imu_topic, args.fix_topic]):
            stamp = t.to_sec()
            if topic == args.odom_topic:
                odom_times.append(stamp)
                odom_x.append(msg.pose.pose.position.x)
                odom_y.append(msg.pose.pose.position.y)
                odom_yaw.append(yaw_from_quaternion(msg.pose.pose.orientation))
                odom_vx.append(msg.twist.twist.linear.x)
                odom_vy.append(msg.twist.twist.linear.y)
                odom_wz.append(msg.twist.twist.angular.z)
            elif topic == args.imu_topic:
                imu_times.append(stamp)
                imu_wz.append(msg.angular_velocity.z)
            elif topic == args.fix_topic:
                if math.isnan(msg.latitude) or math.isnan(msg.longitude):
                    continue
                fix_times.append(stamp)
                fix_lat.append(msg.latitude)
                fix_lon.append(msg.longitude)
                fix_status.append(msg.status.status)

    if gps_offset is None:
        print(
            f"missing /tf_static transform {args.base_frame} -> {args.gps_frame} in bag",
            file=sys.stderr,
        )
        return 1
    if len(odom_times) < 10 or len(fix_times) < 10 or len(imu_times) < 10:
        print("bag does not contain enough odom, RTK, or IMU messages", file=sys.stderr)
        return 1

    lat0 = fix_lat[0]
    lon0 = fix_lon[0]
    gps_x = []
    gps_y = []
    for lat, lon in zip(fix_lat, fix_lon):
        x, y = latlon_to_local_xy(lat, lon, lat0, lon0)
        gps_x.append(x)
        gps_y.append(y)

    off_x, off_y = gps_offset
    base_ref_x: list[float] = []
    base_ref_y: list[float] = []
    matched_odom_points: list[tuple[float, float]] = []
    matched_ref_points: list[tuple[float, float]] = []
    matched_fix_indices: list[int] = []

    for i, stamp in enumerate(fix_times):
        yaw = interpolate_angle(odom_times, odom_yaw, stamp)
        ox = interpolate_scalar(odom_times, odom_x, stamp)
        oy = interpolate_scalar(odom_times, odom_y, stamp)
        if yaw is None or ox is None or oy is None:
            base_ref_x.append(float("nan"))
            base_ref_y.append(float("nan"))
            continue
        c = math.cos(yaw)
        s = math.sin(yaw)
        base_x = gps_x[i] - (c * off_x - s * off_y)
        base_y = gps_y[i] - (s * off_x + c * off_y)
        base_ref_x.append(base_x)
        base_ref_y.append(base_y)
        matched_ref_points.append((base_x, base_y))
        matched_odom_points.append((ox, oy))
        matched_fix_indices.append(i)

    if len(matched_ref_points) < 10:
        print("not enough matched RTK/odom samples to fit a planar transform", file=sys.stderr)
        return 1

    align_theta, align_tx, align_ty = fit_se2(matched_ref_points, matched_odom_points)

    aligned_ref_x: list[float] = []
    aligned_ref_y: list[float] = []
    for x, y in zip(base_ref_x, base_ref_y):
        if math.isnan(x) or math.isnan(y):
            aligned_ref_x.append(float("nan"))
            aligned_ref_y.append(float("nan"))
            continue
        ax, ay = apply_se2(align_theta, align_tx, align_ty, x, y)
        aligned_ref_x.append(ax)
        aligned_ref_y.append(ay)

    ref_vx_world, ref_vy_world, ref_speed, ref_course = compute_rtk_derivatives(
        fix_times, aligned_ref_x, aligned_ref_y
    )

    stationary_flags = [
        abs(vx) < args.stationary_linear_threshold
        and abs(vy) < args.stationary_linear_threshold
        and abs(wz) < args.stationary_angular_threshold
        for vx, vy, wz in zip(odom_vx, odom_vy, odom_wz)
    ]

    stationary_imu_wz = []
    for stamp, wz in zip(imu_times, imu_wz):
        idx = bisect.bisect_right(odom_times, stamp) - 1
        if idx >= 0 and stationary_flags[idx]:
            stationary_imu_wz.append(wz)
    gyro_bias_z = mean(stationary_imu_wz) or 0.0
    imu_wz_bias_corrected = [wz - gyro_bias_z for wz in imu_wz]
    imu_yaw_rel = integrate_imu_yaw(imu_times, imu_wz_bias_corrected)

    yaw_anchor_offsets = []
    for stamp, speed, course in zip(fix_times, ref_speed, ref_course):
        if speed is None or speed < args.min_rtk_speed_for_course or course is None:
            continue
        imu_yaw = interpolate_angle(imu_times, imu_yaw_rel, stamp)
        if imu_yaw is None:
            continue
        yaw_anchor_offsets.append(wrap_angle(course - imu_yaw))
    yaw_offset = circular_mean(yaw_anchor_offsets)
    if yaw_offset is None:
        yaw_offset = 0.0

    residual_x: list[float] = []
    residual_y: list[float] = []
    residual_yaw: list[float] = []
    residual_vx: list[float] = []
    residual_wz: list[float] = []

    yaw_samples = 0
    vx_samples = 0
    wz_samples = 0

    for i, stamp in enumerate(fix_times):
        ox = interpolate_scalar(odom_times, odom_x, stamp)
        oy = interpolate_scalar(odom_times, odom_y, stamp)
        oyaw = interpolate_angle(odom_times, odom_yaw, stamp)
        ovx = interpolate_scalar(odom_times, odom_vx, stamp)
        owz = interpolate_scalar(odom_times, odom_wz, stamp)
        if (
            ox is None
            or oy is None
            or oyaw is None
            or ovx is None
            or owz is None
            or math.isnan(aligned_ref_x[i])
            or math.isnan(aligned_ref_y[i])
        ):
            continue

        residual_x.append(ox - aligned_ref_x[i])
        residual_y.append(oy - aligned_ref_y[i])

        if ref_speed[i] is not None and ref_speed[i] >= args.min_rtk_speed_for_vx:
            ref_vx_body = math.cos(oyaw) * ref_vx_world[i] + math.sin(oyaw) * ref_vy_world[i]
            residual_vx.append(ovx - ref_vx_body)
            vx_samples += 1

        ref_yaw = interpolate_angle(imu_times, imu_yaw_rel, stamp)
        if ref_yaw is not None:
            residual_yaw.append(wrap_angle(oyaw - wrap_angle(ref_yaw + yaw_offset)))
            yaw_samples += 1

    for stamp, owz in zip(odom_times, odom_wz):
        iwz = interpolate_scalar(imu_times, imu_wz_bias_corrected, stamp)
        if iwz is None:
            continue
        if abs(owz) < args.min_turn_rate_for_vyaw and abs(iwz) < args.min_turn_rate_for_vyaw:
            continue
        residual_wz.append(owz - iwz)
        wz_samples += 1

    cov_x = variance(residual_x)
    cov_y = variance(residual_y)
    cov_yaw = variance(residual_yaw)
    cov_vx = variance(residual_vx)
    cov_vyaw = variance(residual_wz)
    cov_vy = cov_vx if args.copy_vy_from_vx else None

    print(f"Bag: {bag_path}")
    print(f"Matched RTK/odom samples: {len(matched_ref_points)}")
    print(
        "Alignment ref->odom: "
        f"yaw={math.degrees(align_theta):.3f} deg "
        f"tx={align_tx:.3f} m ty={align_ty:.3f} m"
    )
    print(
        "Static TF used: "
        f"{args.base_frame} -> {args.gps_frame} "
        f"x={off_x:.3f} m y={off_y:.3f} m"
    )

    print("\nResidual statistics")
    print(
        f"x residual std={fmt(stddev(residual_x), 4)} m "
        f"var={fmt(cov_x, 6)}"
    )
    print(
        f"y residual std={fmt(stddev(residual_y), 4)} m "
        f"var={fmt(cov_y, 6)}"
    )
    print(
        f"yaw residual std={fmt(math.degrees(stddev(residual_yaw)) if residual_yaw else None, 3)} deg "
        f"var={fmt(cov_yaw, 6)} from {yaw_samples} IMU-referenced samples"
    )
    print(
        f"vx residual std={fmt(stddev(residual_vx), 4)} m/s "
        f"var={fmt(cov_vx, 6)} from {vx_samples} RTK speed samples"
    )
    print(
        f"vyaw residual std={fmt(stddev(residual_wz), 4)} rad/s "
        f"var={fmt(cov_vyaw, 6)} from {wz_samples} odom/IMU samples"
    )

    print("\nSuggested wheel odom covariance")
    print("pose_covariance_diagonal:")
    print(f"  x: {fmt(cov_x, 6)}")
    print(f"  y: {fmt(cov_y, 6)}")
    print(f"  yaw: {fmt(cov_yaw, 6)}")
    print("twist_covariance_diagonal:")
    print(f"  vx: {fmt(cov_vx, 6)}")
    if cov_vy is not None:
        print(f"  vy: {fmt(cov_vy, 6)}  # copied from vx")
    else:
        print("  vy: n/a")
    print(f"  vyaw: {fmt(cov_vyaw, 6)}")

    print("\nYAML fragments")
    if None not in (cov_x, cov_y, cov_yaw):
        print(
            "pose_covariance_diagonal: "
            f"[{fmt(cov_x, 6)}, {fmt(cov_y, 6)}, 1000000.0, 1000000.0, 1000000.0, {fmt(cov_yaw, 6)}]"
        )
    if None not in (cov_vx, cov_vy, cov_vyaw):
        print(
            "twist_covariance_diagonal: "
            f"[{fmt(cov_vx, 6)}, {fmt(cov_vy, 6)}, 1000000.0, 1000000.0, 1000000.0, {fmt(cov_vyaw, 6)}]"
        )

    print("\nNotes")
    print("- x/y use RTK position after a 2D rigid alignment into the odom frame.")
    print(
        "- yaw uses IMU wz integration with a heading offset anchored by RTK course-over-ground."
    )
    print("- vyaw uses IMU wz as a short-term reference; this is conservative, not ground truth.")
    print("- vy is copied from vx because this bag has limited lateral excitation.")
    print(
        f"- IMU yaw-rate bias removed before integration: wz_bias={gyro_bias_z:.6f} rad/s "
        f"from {len(stationary_imu_wz)} stationary IMU samples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

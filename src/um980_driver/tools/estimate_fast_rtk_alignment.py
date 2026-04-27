#!/usr/bin/env python3
"""Estimate the planar transform from RTK map into FAST localization map.

This tool aligns two trajectories recorded in the same bag:

  - FAST-LOCALIZATION odometry, typically /Odometry in the lidar map frame
  - RTK global odometry, typically /odometry/global in the rtk_map frame

It fits a 2D rigid transform T_map_rtk_map such that:

  p_map ~= T_map_rtk_map * p_rtk_map

The output is intended to be used as a static TF between ``map`` and
``rtk_map`` after verifying the residuals are acceptably small.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics
import sys
from collections import deque

import rosbag


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


def quaternion_tuple_to_yaw(
    x: float, y: float, z: float, w: float
) -> float:
    class QuaternionLike:
        def __init__(self, qx: float, qy: float, qz: float, qw: float) -> None:
            self.x = qx
            self.y = qy
            self.z = qz
            self.w = qw

    return yaw_from_quaternion(QuaternionLike(x, y, z, w))


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def path_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for index in range(1, len(points)):
        total += math.hypot(
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
    return total


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


def fit_se2(
    ref_points: list[tuple[float, float]], target_points: list[tuple[float, float]]
) -> tuple[float, float, float]:
    if len(ref_points) != len(target_points) or len(ref_points) < 2:
        raise ValueError("need at least two matched points for SE(2) fit")

    ref_cx = statistics.fmean(point[0] for point in ref_points)
    ref_cy = statistics.fmean(point[1] for point in ref_points)
    target_cx = statistics.fmean(point[0] for point in target_points)
    target_cy = statistics.fmean(point[1] for point in target_points)

    cross = 0.0
    dot = 0.0
    for (rx, ry), (tx, ty) in zip(ref_points, target_points):
        rcx = rx - ref_cx
        rcy = ry - ref_cy
        tcx = tx - target_cx
        tcy = ty - target_cy
        dot += rcx * tcx + rcy * tcy
        cross += rcx * tcy - rcy * tcx

    theta = math.atan2(cross, dot)
    c = math.cos(theta)
    s = math.sin(theta)
    tx = target_cx - (c * ref_cx - s * ref_cy)
    ty = target_cy - (s * ref_cx + c * ref_cy)
    return theta, tx, ty


def apply_se2(theta: float, tx: float, ty: float, x: float, y: float) -> tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return c * x - s * y + tx, s * x + c * y + ty


def compose_se2(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> tuple[float, float, float]:
    yaw1, x1, y1 = first
    yaw2, x2, y2 = second
    c = math.cos(yaw1)
    s = math.sin(yaw1)
    x = c * x2 - s * y2 + x1
    y = s * x2 + c * y2 + y1
    return wrap_angle(yaw1 + yaw2), x, y


def inverse_se2(transform: tuple[float, float, float]) -> tuple[float, float, float]:
    yaw, x, y = transform
    c = math.cos(yaw)
    s = math.sin(yaw)
    inv_yaw = wrap_angle(-yaw)
    inv_x = -(c * x + s * y)
    inv_y = s * x - c * y
    return inv_yaw, inv_x, inv_y


def stamp_to_sec(msg, bag_stamp: float) -> float:
    if hasattr(msg, "header"):
        stamp = msg.header.stamp.to_sec()
        if stamp > 0.0:
            return stamp
    return bag_stamp


def extract_static_transforms(
    bag: rosbag.Bag,
) -> tuple[dict[str, list[tuple[str, tuple[float, float, float]]]], list[str]]:
    adjacency: dict[str, list[tuple[str, tuple[float, float, float]]]] = {}
    warnings: list[str] = []

    for _, msg, _ in bag.read_messages(topics=["/tf_static"]):
        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            yaw = quaternion_tuple_to_yaw(
                rotation.x, rotation.y, rotation.z, rotation.w
            )
            roll_pitch_magnitude = math.hypot(rotation.x, rotation.y)
            if abs(translation.z) > 1e-4 or roll_pitch_magnitude > 1e-4:
                warnings.append(
                    "Ignoring z/roll/pitch in static TF "
                    f"{parent} -> {child} (z={translation.z:.4f})"
                )
            se2 = (yaw, float(translation.x), float(translation.y))
            adjacency.setdefault(parent, []).append((child, se2))
            adjacency.setdefault(child, []).append((parent, inverse_se2(se2)))

    return adjacency, warnings


def find_static_transform(
    adjacency: dict[str, list[tuple[str, tuple[float, float, float]]]],
    source_frame: str,
    target_frame: str,
) -> tuple[float, float, float] | None:
    if source_frame == target_frame:
        return (0.0, 0.0, 0.0)

    queue: deque[tuple[str, tuple[float, float, float]]] = deque()
    visited = {source_frame}
    queue.append((source_frame, (0.0, 0.0, 0.0)))

    while queue:
        frame, transform = queue.popleft()
        for next_frame, edge in adjacency.get(frame, []):
            if next_frame in visited:
                continue
            composed = compose_se2(transform, edge)
            if next_frame == target_frame:
                return composed
            visited.add(next_frame)
            queue.append((next_frame, composed))

    return None


def maybe_reframe_track(
    name: str,
    child_frame: str,
    target_frame: str,
    times: list[float],
    xs: list[float],
    ys: list[float],
    yaws: list[float],
    adjacency: dict[str, list[tuple[str, tuple[float, float, float]]]],
    assume_body_is_base_link: bool,
) -> tuple[str, list[float], list[float], list[float], str | None]:
    if child_frame == target_frame:
        return child_frame, xs, ys, yaws, None

    static_tf = find_static_transform(adjacency, child_frame, target_frame)
    if static_tf is None:
        if assume_body_is_base_link and {child_frame, target_frame} == {"body", "base_link"}:
            note = (
                f"{name} reframed from '{child_frame}' to '{target_frame}' using "
                "explicit --assume-body-is-base-link identity assumption"
            )
            return target_frame, xs, ys, yaws, note
        raise RuntimeError(
            f"{name} child_frame is '{child_frame}', but no static TF to '{target_frame}' "
            "was found in /tf_static. If this is the FAST body/base_link naming "
            "case and those frames are physically identical, rerun with "
            "--assume-body-is-base-link."
        )

    tf_yaw, tf_x, tf_y = static_tf
    reframed_xs: list[float] = []
    reframed_ys: list[float] = []
    reframed_yaws: list[float] = []

    for _, x, y, yaw in zip(times, xs, ys, yaws):
        c = math.cos(yaw)
        s = math.sin(yaw)
        reframed_xs.append(x + c * tf_x - s * tf_y)
        reframed_ys.append(y + s * tf_x + c * tf_y)
        reframed_yaws.append(wrap_angle(yaw + tf_yaw))

    note = (
        f"{name} reframed from '{child_frame}' to '{target_frame}' using "
        f"static TF dx={tf_x:.3f} dy={tf_y:.3f} dyaw={math.degrees(tf_yaw):.3f} deg"
    )
    return target_frame, reframed_xs, reframed_ys, reframed_yaws, note


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the planar transform from RTK odom frame into the FAST "
            "localization map frame from a rosbag."
        )
    )
    parser.add_argument("bag", help="Path to the rosbag file")
    parser.add_argument("--fast-odom-topic", default="/Odometry")
    parser.add_argument("--rtk-odom-topic", default="/odometry/global")
    parser.add_argument(
        "--quality-topic",
        default="/rtk/fix_type",
        help="Optional std_msgs/String topic used to keep only trustworthy RTK samples",
    )
    parser.add_argument(
        "--accepted-quality",
        nargs="+",
        default=["rtk_fixed"],
        help="Accepted quality labels from --quality-topic",
    )
    parser.add_argument(
        "--disable-quality-filter",
        action="store_true",
        help="Use all RTK odom samples regardless of the quality topic",
    )
    parser.add_argument(
        "--max-quality-age",
        type=float,
        default=1.0,
        help="Maximum age in seconds between an RTK odom sample and the latest quality label",
    )
    parser.add_argument(
        "--target-base-frame",
        default="base_link",
        help="Convert both trajectories to this child frame before alignment",
    )
    parser.add_argument(
        "--assume-body-is-base-link",
        action="store_true",
        help=(
            "Treat FAST 'body' and robot 'base_link' as an identity transform when "
            "the bag does not contain that static TF. Use only if they refer to the "
            "same physical point."
        ),
    )
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--rtk-map-frame", default="rtk_map")
    parser.add_argument(
        "--min-sample-distance",
        type=float,
        default=0.5,
        help="Minimum RTK travel distance between accepted match samples",
    )
    parser.add_argument(
        "--skip-first-seconds",
        type=float,
        default=3.0,
        help="Ignore the first seconds after bag start to skip initialization transients",
    )
    parser.add_argument(
        "--max-residual",
        type=float,
        default=0.0,
        help="Optional residual threshold in meters for a second fitting pass; 0 disables it",
    )
    parser.add_argument(
        "--export-json",
        default="",
        help="Optional JSON path for matched trajectories; default is derived from the bag name",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        print(f"bag file not found: {bag_path}", file=sys.stderr)
        return 1

    accepted_quality = {item.strip().lower() for item in args.accepted_quality if item.strip()}
    if not accepted_quality and not args.disable_quality_filter:
        print("accepted quality labels are empty", file=sys.stderr)
        return 1

    fast_times: list[float] = []
    fast_xs: list[float] = []
    fast_ys: list[float] = []
    fast_yaws: list[float] = []
    fast_header_frames: set[str] = set()
    fast_child_frames: set[str] = set()

    rtk_times: list[float] = []
    rtk_xs: list[float] = []
    rtk_ys: list[float] = []
    rtk_yaws: list[float] = []
    rtk_header_frames: set[str] = set()
    rtk_child_frames: set[str] = set()

    quality_times: list[float] = []
    quality_labels: list[str] = []

    with rosbag.Bag(bag_path, "r") as bag:
        bag_start = bag.get_start_time()
        bag_end = bag.get_end_time()
        adjacency, tf_warnings = extract_static_transforms(bag)

        topics = [args.fast_odom_topic, args.rtk_odom_topic]
        if not args.disable_quality_filter and args.quality_topic:
            topics.append(args.quality_topic)

        for topic, msg, bag_stamp in bag.read_messages(topics=topics):
            stamp = stamp_to_sec(msg, bag_stamp.to_sec())

            if topic == args.fast_odom_topic:
                fast_times.append(stamp)
                fast_xs.append(msg.pose.pose.position.x)
                fast_ys.append(msg.pose.pose.position.y)
                fast_yaws.append(yaw_from_quaternion(msg.pose.pose.orientation))
                if msg.header.frame_id:
                    fast_header_frames.add(msg.header.frame_id)
                if msg.child_frame_id:
                    fast_child_frames.add(msg.child_frame_id)

            elif topic == args.rtk_odom_topic:
                rtk_times.append(stamp)
                rtk_xs.append(msg.pose.pose.position.x)
                rtk_ys.append(msg.pose.pose.position.y)
                rtk_yaws.append(yaw_from_quaternion(msg.pose.pose.orientation))
                if msg.header.frame_id:
                    rtk_header_frames.add(msg.header.frame_id)
                if msg.child_frame_id:
                    rtk_child_frames.add(msg.child_frame_id)

            elif topic == args.quality_topic:
                quality = getattr(msg, "data", "")
                if not isinstance(quality, str):
                    continue
                quality = quality.strip().lower()
                if quality:
                    quality_times.append(stamp)
                    quality_labels.append(quality)

    if len(fast_times) < 10:
        print(f"not enough FAST odom messages on {args.fast_odom_topic}", file=sys.stderr)
        return 1
    if len(rtk_times) < 10:
        print(f"not enough RTK odom messages on {args.rtk_odom_topic}", file=sys.stderr)
        return 1

    fast_child_frame = next(iter(fast_child_frames)) if fast_child_frames else ""
    rtk_child_frame = next(iter(rtk_child_frames)) if rtk_child_frames else ""
    if len(fast_child_frames) > 1 or len(rtk_child_frames) > 1:
        print("odom child frames changed within the bag, please record a clean bag", file=sys.stderr)
        return 1

    try:
        (
            fast_child_frame,
            fast_xs,
            fast_ys,
            fast_yaws,
            fast_reframe_note,
        ) = maybe_reframe_track(
            "FAST",
            fast_child_frame or args.target_base_frame,
            args.target_base_frame,
            fast_times,
            fast_xs,
            fast_ys,
            fast_yaws,
            adjacency,
            args.assume_body_is_base_link,
        )
        (
            rtk_child_frame,
            rtk_xs,
            rtk_ys,
            rtk_yaws,
            rtk_reframe_note,
        ) = maybe_reframe_track(
            "RTK",
            rtk_child_frame or args.target_base_frame,
            args.target_base_frame,
            rtk_times,
            rtk_xs,
            rtk_ys,
            rtk_yaws,
            adjacency,
            args.assume_body_is_base_link,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    overlap_start = max(fast_times[0], rtk_times[0], bag_start + max(args.skip_first_seconds, 0.0))
    overlap_end = min(fast_times[-1], rtk_times[-1], bag_end)
    if overlap_end <= overlap_start:
        print("FAST and RTK trajectories do not overlap in time", file=sys.stderr)
        return 1

    matched_times: list[float] = []
    matched_rtk_points: list[tuple[float, float]] = []
    matched_fast_points: list[tuple[float, float]] = []
    matched_quality_labels: list[str] = []

    last_kept_rtk_point: tuple[float, float] | None = None

    for stamp, rtk_x, rtk_y in zip(rtk_times, rtk_xs, rtk_ys):
        if stamp < overlap_start or stamp > overlap_end:
            continue

        quality_label = ""
        if not args.disable_quality_filter and quality_times:
            idx = bisect.bisect_right(quality_times, stamp) - 1
            if idx < 0:
                continue
            quality_label = quality_labels[idx]
            age = stamp - quality_times[idx]
            if age > args.max_quality_age:
                continue
            if quality_label not in accepted_quality:
                continue

        fast_x = interpolate_scalar(fast_times, fast_xs, stamp)
        fast_y = interpolate_scalar(fast_times, fast_ys, stamp)
        if fast_x is None or fast_y is None:
            continue

        current_rtk_point = (rtk_x, rtk_y)
        if last_kept_rtk_point is not None:
            delta = math.hypot(
                current_rtk_point[0] - last_kept_rtk_point[0],
                current_rtk_point[1] - last_kept_rtk_point[1],
            )
            if delta < args.min_sample_distance:
                continue

        matched_times.append(stamp)
        matched_rtk_points.append(current_rtk_point)
        matched_fast_points.append((fast_x, fast_y))
        matched_quality_labels.append(quality_label)
        last_kept_rtk_point = current_rtk_point

    if len(matched_rtk_points) < 6:
        print(
            "not enough matched samples after filtering; try recording a longer route "
            "with more turning and stable RTK",
            file=sys.stderr,
        )
        return 1

    yaw_map_from_rtk, tx_map_from_rtk, ty_map_from_rtk = fit_se2(
        matched_rtk_points, matched_fast_points
    )

    aligned_rtk_points = [
        apply_se2(yaw_map_from_rtk, tx_map_from_rtk, ty_map_from_rtk, x, y)
        for x, y in matched_rtk_points
    ]
    residuals = [
        math.hypot(aligned_x - fast_x, aligned_y - fast_y)
        for (aligned_x, aligned_y), (fast_x, fast_y) in zip(
            aligned_rtk_points, matched_fast_points
        )
    ]

    if args.max_residual > 0.0:
        filtered_rtk_points: list[tuple[float, float]] = []
        filtered_fast_points: list[tuple[float, float]] = []
        filtered_times: list[float] = []
        filtered_quality: list[str] = []
        for stamp, quality, residual, rtk_point, fast_point in zip(
            matched_times,
            matched_quality_labels,
            residuals,
            matched_rtk_points,
            matched_fast_points,
        ):
            if residual <= args.max_residual:
                filtered_times.append(stamp)
                filtered_quality.append(quality)
                filtered_rtk_points.append(rtk_point)
                filtered_fast_points.append(fast_point)
        if len(filtered_rtk_points) >= 6 and len(filtered_rtk_points) < len(matched_rtk_points):
            matched_times = filtered_times
            matched_quality_labels = filtered_quality
            matched_rtk_points = filtered_rtk_points
            matched_fast_points = filtered_fast_points
            yaw_map_from_rtk, tx_map_from_rtk, ty_map_from_rtk = fit_se2(
                matched_rtk_points, matched_fast_points
            )
            aligned_rtk_points = [
                apply_se2(yaw_map_from_rtk, tx_map_from_rtk, ty_map_from_rtk, x, y)
                for x, y in matched_rtk_points
            ]
            residuals = [
                math.hypot(aligned_x - fast_x, aligned_y - fast_y)
                for (aligned_x, aligned_y), (fast_x, fast_y) in zip(
                    aligned_rtk_points, matched_fast_points
                )
            ]

    rmse = math.sqrt(statistics.fmean(residual * residual for residual in residuals))
    mean_residual = statistics.fmean(residuals)
    median_residual = statistics.median(residuals)
    max_residual = max(residuals)

    export_path = args.export_json
    if not export_path:
        root, ext = os.path.splitext(bag_path)
        export_path = root + "_fast_rtk_alignment.json"

    export_payload = {
        "bag": bag_path,
        "fast_odom_topic": args.fast_odom_topic,
        "rtk_odom_topic": args.rtk_odom_topic,
        "quality_topic": "" if args.disable_quality_filter else args.quality_topic,
        "accepted_quality": sorted(accepted_quality),
        "transform_map_from_rtk_map": {
            "yaw_rad": yaw_map_from_rtk,
            "yaw_deg": math.degrees(yaw_map_from_rtk),
            "tx_m": tx_map_from_rtk,
            "ty_m": ty_map_from_rtk,
        },
        "residual_summary": {
            "mean_m": mean_residual,
            "median_m": median_residual,
            "rmse_m": rmse,
            "max_m": max_residual,
        },
        "matched_samples": {
            "time_sec": matched_times,
            "rtk_xy": matched_rtk_points,
            "fast_xy": matched_fast_points,
            "aligned_rtk_xy": aligned_rtk_points,
            "residual_m": residuals,
            "quality": matched_quality_labels,
        },
    }

    with open(export_path, "w", encoding="utf-8") as output_file:
        json.dump(export_payload, output_file, indent=2)

    print(f"Bag: {bag_path}")
    print(f"Overlap window: {fmt(overlap_start, 3)} -> {fmt(overlap_end, 3)} s")
    print(f"FAST topic: {args.fast_odom_topic}")
    print(f"RTK topic:  {args.rtk_odom_topic}")
    print(
        f"FAST frames: header={sorted(fast_header_frames)} child={sorted(fast_child_frames)}"
    )
    print(
        f"RTK frames:  header={sorted(rtk_header_frames)} child={sorted(rtk_child_frames)}"
    )
    print(f"Target base frame: {args.target_base_frame}")
    for warning in tf_warnings:
        print(f"TF note: {warning}")
    if fast_reframe_note:
        print(f"Reframe: {fast_reframe_note}")
    if rtk_reframe_note:
        print(f"Reframe: {rtk_reframe_note}")

    print("\nMatched trajectory summary")
    print(f"Samples: {len(matched_times)}")
    print(f"FAST path length: {fmt(path_length(matched_fast_points), 3)} m")
    print(f"RTK path length:  {fmt(path_length(matched_rtk_points), 3)} m")
    if not args.disable_quality_filter:
        quality_counter: dict[str, int] = {}
        for label in matched_quality_labels:
            quality_counter[label] = quality_counter.get(label, 0) + 1
        print(f"Accepted quality labels: {quality_counter}")

    print("\nEstimated transform T_map_rtk_map")
    print(f"tx:      {fmt(tx_map_from_rtk, 4)} m")
    print(f"ty:      {fmt(ty_map_from_rtk, 4)} m")
    print(
        f"yaw:     {fmt(yaw_map_from_rtk, 6)} rad  ({fmt(math.degrees(yaw_map_from_rtk), 3)} deg)"
    )

    print("\nResiduals after applying T_map_rtk_map")
    print(f"mean:    {fmt(mean_residual, 4)} m")
    print(f"median:  {fmt(median_residual, 4)} m")
    print(f"rmse:    {fmt(rmse, 4)} m")
    print(f"max:     {fmt(max_residual, 4)} m")

    print("\nSuggested static TF")
    print(
        "rosrun tf2_ros static_transform_publisher "
        f"{tx_map_from_rtk:.6f} {ty_map_from_rtk:.6f} 0 "
        f"{yaw_map_from_rtk:.6f} 0 0 {args.map_frame} {args.rtk_map_frame}"
    )

    print("\nYAML snippet")
    print(f"{args.map_frame}_from_{args.rtk_map_frame}:")
    print(f"  x: {tx_map_from_rtk:.6f}")
    print(f"  y: {ty_map_from_rtk:.6f}")
    print("  z: 0.0")
    print(f"  yaw: {yaw_map_from_rtk:.6f}")

    print(f"\nExported matched trajectories to {export_path}")

    if rmse > 0.50:
        print(
            "\nWarning: residuals are still fairly large. "
            "Check that FAST and RTK both stayed stable, and verify the child frames "
            "really refer to the same physical point.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

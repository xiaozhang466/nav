#!/usr/bin/env python3
"""Generate navigation waypoints from a recorded pose trajectory.

The first patrol workflow uses this as an offline step:

  pose trajectory in map frame -> evenly spaced move_base goal waypoints

The parser intentionally accepts both proper JSON and the common FAST-LIO/
FAST-LOCALIZATION text pose format where each line is:

  x y z qw qx qy qz
"""

import argparse
import datetime as _datetime
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class PoseSample:
    x: float
    y: float
    z: float
    yaw: Optional[float]
    source_index: int


@dataclass
class Waypoint:
    waypoint_id: int
    x: float
    y: float
    z: float
    yaw: float
    distance: float
    source_index: int


@dataclass
class CornerSampling:
    interval: float
    yaw_threshold_deg: float
    expand_distance: float
    zones: List[Tuple[float, float]]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def unwrap_to_near(angle: float, reference: float) -> float:
    return reference + normalize_angle(angle - reference)


def interpolate_angle(a: float, b: float, ratio: float) -> float:
    return normalize_angle(a + normalize_angle(b - a) * ratio)


def make_pose_from_numbers(
    values: Sequence[float], source_index: int, quat_order: str
) -> PoseSample:
    if len(values) < 2:
        raise ValueError("pose row must contain at least x and y")

    x = float(values[0])
    y = float(values[1])
    z = float(values[2]) if len(values) >= 3 else 0.0
    yaw = None

    if len(values) >= 7:
        if quat_order == "wxyz":
            qw, qx, qy, qz = values[3], values[4], values[5], values[6]
        else:
            qx, qy, qz, qw = values[3], values[4], values[5], values[6]
        yaw = yaw_from_quaternion(float(qx), float(qy), float(qz), float(qw))
    elif len(values) >= 4:
        yaw = float(values[3])

    return PoseSample(x=x, y=y, z=z, yaw=yaw, source_index=source_index)


def find_numeric_sequence(obj) -> Optional[List]:
    if isinstance(obj, list):
        if not obj:
            return obj
        if all(isinstance(item, (list, tuple, dict)) for item in obj):
            return obj
    if isinstance(obj, dict):
        preferred_keys = (
            "poses",
            "trajectory",
            "samples",
            "path",
            "data",
            "positions",
        )
        for key in preferred_keys:
            value = obj.get(key)
            found = find_numeric_sequence(value)
            if found is not None:
                return found
        for value in obj.values():
            found = find_numeric_sequence(value)
            if found is not None:
                return found
    return None


def numbers_from_json_pose(item) -> List[float]:
    if isinstance(item, (list, tuple)):
        return [float(value) for value in item]

    if not isinstance(item, dict):
        raise ValueError(f"unsupported JSON pose item: {type(item).__name__}")

    if all(key in item for key in ("x", "y")):
        values = [float(item["x"]), float(item["y"]), float(item.get("z", 0.0))]
        if "yaw" in item:
            values.append(float(item["yaw"]))
            return values
        if all(key in item for key in ("qx", "qy", "qz", "qw")):
            values.extend(
                [
                    float(item["qx"]),
                    float(item["qy"]),
                    float(item["qz"]),
                    float(item["qw"]),
                ]
            )
            return values
        return values

    pose = item.get("pose", item)
    position = pose.get("position") if isinstance(pose, dict) else None
    orientation = pose.get("orientation") if isinstance(pose, dict) else None
    if isinstance(position, dict):
        values = [
            float(position["x"]),
            float(position["y"]),
            float(position.get("z", 0.0)),
        ]
        if isinstance(orientation, dict):
            values.extend(
                [
                    float(orientation["x"]),
                    float(orientation["y"]),
                    float(orientation["z"]),
                    float(orientation["w"]),
                ]
            )
        return values

    raise ValueError(f"cannot extract x/y from JSON pose item: {item}")


def read_pose_samples(path: str, quat_order: str) -> List[PoseSample]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()

    stripped = raw.lstrip()
    samples: List[PoseSample] = []

    if stripped.startswith("[") or stripped.startswith("{"):
        data = json.loads(raw)
        items = find_numeric_sequence(data)
        if items is None:
            raise ValueError("no pose sequence found in JSON file")
        for index, item in enumerate(items):
            values = numbers_from_json_pose(item)
            # JSON quaternions are usually xyzw; keep the command line switch for
            # explicit formats but default text files can still use wxyz.
            order = "xyzw" if isinstance(item, dict) and len(values) >= 7 else quat_order
            samples.append(make_pose_from_numbers(values, index, order))
        return samples

    for line_no, line in enumerate(raw.splitlines(), 1):
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        parts = clean.replace(",", " ").split()
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"line {line_no}: cannot parse numeric pose row") from exc
        try:
            samples.append(make_pose_from_numbers(values, line_no, quat_order))
        except ValueError as exc:
            raise ValueError(f"line {line_no}: {exc}") from exc

    return samples


def filter_samples(samples: Sequence[PoseSample], min_step: float) -> List[PoseSample]:
    if not samples:
        return []

    filtered = [samples[0]]
    for sample in samples[1:]:
        last = filtered[-1]
        if math.hypot(sample.x - last.x, sample.y - last.y) >= min_step:
            filtered.append(sample)

    if filtered[-1] is not samples[-1]:
        filtered.append(samples[-1])
    return filtered


def cumulative_distances(samples: Sequence[PoseSample]) -> List[float]:
    distances = [0.0]
    total = 0.0
    for previous, current in zip(samples, samples[1:]):
        total += math.hypot(current.x - previous.x, current.y - previous.y)
        distances.append(total)
    return distances


def interpolate_pose_at(
    samples: Sequence[PoseSample], distances: Sequence[float], target: float
) -> Tuple[PoseSample, int]:
    if target <= 0.0:
        return samples[0], 0
    if target >= distances[-1]:
        return samples[-1], len(samples) - 1

    lo = 0
    hi = len(distances) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if distances[mid] < target:
            lo = mid + 1
        else:
            hi = mid

    right = lo
    left = max(0, right - 1)
    segment = max(1e-9, distances[right] - distances[left])
    ratio = (target - distances[left]) / segment

    a = samples[left]
    b = samples[right]
    yaw = None
    if a.yaw is not None and b.yaw is not None:
        yaw = interpolate_angle(a.yaw, b.yaw, ratio)

    return (
        PoseSample(
            x=a.x + (b.x - a.x) * ratio,
            y=a.y + (b.y - a.y) * ratio,
            z=a.z + (b.z - a.z) * ratio,
            yaw=yaw,
            source_index=b.source_index,
        ),
        right,
    )


def tangent_yaw_at(
    samples: Sequence[PoseSample],
    distances: Sequence[float],
    target: float,
    lookahead: float,
) -> float:
    total = distances[-1]
    if total <= 0.0:
        return 0.0

    start_distance = max(0.0, target - lookahead * 0.5)
    end_distance = min(total, target + lookahead * 0.5)
    if end_distance - start_distance < 1e-6:
        start_distance = max(0.0, target - lookahead)
        end_distance = min(total, target + lookahead)
    if end_distance - start_distance < 1e-6:
        return 0.0

    start_pose, _ = interpolate_pose_at(samples, distances, start_distance)
    end_pose, _ = interpolate_pose_at(samples, distances, end_distance)
    return math.atan2(end_pose.y - start_pose.y, end_pose.x - start_pose.x)


def build_sample_distances(total_length: float, interval: float, include_end: bool) -> List[float]:
    if interval <= 0.0:
        raise ValueError("interval must be > 0")

    values = []
    current = 0.0
    while current <= total_length + 1e-9:
        values.append(min(current, total_length))
        current += interval

    if include_end and values and total_length - values[-1] > interval * 0.25:
        values.append(total_length)
    elif include_end and not values:
        values.append(total_length)

    return values


def yaw_between_distances(
    samples: Sequence[PoseSample],
    distances: Sequence[float],
    start_distance: float,
    end_distance: float,
) -> Optional[float]:
    if end_distance - start_distance < 1e-6:
        return None

    start_pose, _ = interpolate_pose_at(samples, distances, start_distance)
    end_pose, _ = interpolate_pose_at(samples, distances, end_distance)
    dx = end_pose.x - start_pose.x
    dy = end_pose.y - start_pose.y
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


def merge_distance_zones(zones: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(zones):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + 1e-6:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def detect_corner_zones(
    samples: Sequence[PoseSample],
    distances: Sequence[float],
    yaw_threshold_deg: float,
    expand_distance: float,
    scan_step: float,
) -> List[Tuple[float, float]]:
    if yaw_threshold_deg <= 0.0:
        raise ValueError("corner yaw threshold must be > 0")
    if expand_distance <= 0.0:
        raise ValueError("corner expand distance must be > 0")
    if scan_step <= 0.0:
        raise ValueError("corner scan step must be > 0")

    total_length = distances[-1]
    yaw_threshold = math.radians(yaw_threshold_deg)
    probe = min(expand_distance, max(0.1, total_length * 0.5))
    zones: List[Tuple[float, float]] = []

    current = 0.0
    while current <= total_length + 1e-9:
        before_yaw = yaw_between_distances(
            samples,
            distances,
            max(0.0, current - probe),
            current,
        )
        after_yaw = yaw_between_distances(
            samples,
            distances,
            current,
            min(total_length, current + probe),
        )
        if before_yaw is not None and after_yaw is not None:
            turn_angle = abs(normalize_angle(after_yaw - before_yaw))
            if turn_angle >= yaw_threshold:
                zones.append(
                    (
                        max(0.0, current - expand_distance),
                        min(total_length, current + expand_distance),
                    )
                )
        current += scan_step

    return merge_distance_zones(zones)


def unique_sorted_distances(values: Sequence[float]) -> List[float]:
    result: List[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > 1e-6:
            result.append(value)
    return result


def build_corner_aware_sample_distances(
    total_length: float,
    interval: float,
    include_end: bool,
    corner_zones: Sequence[Tuple[float, float]],
    corner_interval: float,
) -> List[float]:
    if not corner_zones:
        return build_sample_distances(total_length, interval, include_end)
    if corner_interval <= 0.0:
        raise ValueError("corner interval must be > 0")
    if corner_interval >= interval:
        raise ValueError("corner interval must be smaller than base interval")

    values = build_sample_distances(total_length, interval, include_end)
    for start, end in corner_zones:
        current = max(0.0, start)
        end = min(total_length, end)
        while current <= end + 1e-9:
            values.append(current)
            current += corner_interval
        values.append(end)

    return unique_sorted_distances(values)


def generate_waypoints(
    samples: Sequence[PoseSample],
    interval: float,
    include_end: bool,
    yaw_mode: str,
    heading_lookahead: float,
    corner_interval: float,
    corner_yaw_threshold_deg: float,
    corner_expand_distance: float,
) -> Tuple[List[Waypoint], float, CornerSampling]:
    if len(samples) < 2:
        raise ValueError("need at least two pose samples")

    distances = cumulative_distances(samples)
    total_length = distances[-1]
    if total_length <= 0.0:
        raise ValueError("trajectory length is zero")

    corner_zones: List[Tuple[float, float]] = []
    if corner_interval > 0.0:
        scan_step = max(0.25, min(corner_interval, interval, corner_expand_distance * 0.25))
        corner_zones = detect_corner_zones(
            samples,
            distances,
            yaw_threshold_deg=corner_yaw_threshold_deg,
            expand_distance=corner_expand_distance,
            scan_step=scan_step,
        )
        sample_distances = build_corner_aware_sample_distances(
            total_length,
            interval,
            include_end,
            corner_zones,
            corner_interval,
        )
    else:
        sample_distances = build_sample_distances(total_length, interval, include_end)

    waypoints = []
    previous_yaw = None
    for waypoint_id, distance in enumerate(sample_distances):
        pose, source_index = interpolate_pose_at(samples, distances, distance)
        if yaw_mode == "pose" and pose.yaw is not None:
            yaw = pose.yaw
        else:
            yaw = tangent_yaw_at(samples, distances, distance, heading_lookahead)
        if previous_yaw is not None:
            yaw = unwrap_to_near(yaw, previous_yaw)
        previous_yaw = yaw
        waypoints.append(
            Waypoint(
                waypoint_id=waypoint_id,
                x=pose.x,
                y=pose.y,
                z=pose.z,
                yaw=normalize_angle(yaw),
                distance=distance,
                source_index=source_index,
            )
        )

    return (
        waypoints,
        total_length,
        CornerSampling(
            interval=corner_interval,
            yaw_threshold_deg=corner_yaw_threshold_deg,
            expand_distance=corner_expand_distance,
            zones=corner_zones,
        ),
    )


def format_distance_zones(zones: Sequence[Tuple[float, float]]) -> str:
    if not zones:
        return "[]"
    items = [f"[{start:.3f}, {end:.3f}]" for start, end in zones]
    return "[" + ", ".join(items) + "]"


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_waypoints_yaml(
    path: str,
    waypoints: Sequence[Waypoint],
    frame_id: str,
    source_path: str,
    raw_count: int,
    filtered_count: int,
    interval: float,
    path_length: float,
    loop: bool,
    yaw_mode: str,
    corner_sampling: CornerSampling,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    generated_at = _datetime.datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"frame_id: {yaml_quote(frame_id)}\n")
        handle.write(f"loop: {'true' if loop else 'false'}\n")
        handle.write(f"source: {yaml_quote(os.path.abspath(source_path))}\n")
        handle.write(f"generated_at: {yaml_quote(generated_at)}\n")
        handle.write(f"raw_pose_count: {raw_count}\n")
        handle.write(f"filtered_pose_count: {filtered_count}\n")
        handle.write(f"interval: {interval:.6f}\n")
        if corner_sampling.interval > 0.0:
            handle.write(f"corner_interval: {corner_sampling.interval:.6f}\n")
            handle.write(
                f"corner_yaw_threshold_deg: {corner_sampling.yaw_threshold_deg:.3f}\n"
            )
            handle.write(f"corner_expand_distance: {corner_sampling.expand_distance:.6f}\n")
            handle.write(f"corner_zone_count: {len(corner_sampling.zones)}\n")
        handle.write(f"path_length: {path_length:.6f}\n")
        handle.write(f"yaw_mode: {yaml_quote(yaw_mode)}\n")
        handle.write("waypoints:\n")
        for waypoint in waypoints:
            handle.write(f"  - id: {waypoint.waypoint_id}\n")
            handle.write(f"    x: {waypoint.x:.6f}\n")
            handle.write(f"    y: {waypoint.y:.6f}\n")
            handle.write(f"    z: {waypoint.z:.6f}\n")
            handle.write(f"    yaw: {waypoint.yaw:.6f}\n")
            handle.write(f"    yaw_deg: {math.degrees(waypoint.yaw):.3f}\n")
            handle.write(f"    distance: {waypoint.distance:.6f}\n")
            handle.write(f"    source_index: {waypoint.source_index}\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample evenly spaced navigation waypoints from a recorded pose trajectory."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="src/um980_driver/poses/pose.json",
        help="Input pose trajectory file. Supports JSON or whitespace rows: x y z qw qx qy qz.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="src/um980_driver/waypoints/inspection_waypoints.yaml",
        help="Output waypoints YAML file.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Distance between generated waypoints in meters.",
    )
    parser.add_argument("--frame-id", default="map", help="Waypoint frame id.")
    parser.add_argument(
        "--quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Quaternion order for whitespace pose rows.",
    )
    parser.add_argument(
        "--min-input-step",
        type=float,
        default=0.05,
        help="Ignore consecutive raw pose samples closer than this distance in meters.",
    )
    parser.add_argument(
        "--yaw-mode",
        choices=("tangent", "pose"),
        default="tangent",
        help="Use route tangent yaw or recorded pose yaw for waypoint orientation.",
    )
    parser.add_argument(
        "--heading-lookahead",
        type=float,
        default=0.8,
        help="Distance window used to estimate tangent heading.",
    )
    parser.add_argument(
        "--corner-interval",
        type=float,
        default=0.5,
        help="Waypoint interval inside detected sharp-turn zones in meters. Set <=0 to disable.",
    )
    parser.add_argument(
        "--corner-yaw-threshold-deg",
        type=float,
        default=35.0,
        help="Minimum heading change used to detect a sharp turn.",
    )
    parser.add_argument(
        "--corner-expand-distance",
        type=float,
        default=4.0,
        help="Distance before and after each detected turn to sample with --corner-interval.",
    )
    parser.add_argument(
        "--no-include-end",
        action="store_true",
        help="Do not force-add the final trajectory point when it is far from the last interval sample.",
    )
    parser.add_argument("--loop", action="store_true", help="Mark the waypoint route as cyclic.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        samples = read_pose_samples(args.input, args.quat_order)
        filtered = filter_samples(samples, args.min_input_step)
        waypoints, path_length, corner_sampling = generate_waypoints(
            filtered,
            interval=args.interval,
            include_end=not args.no_include_end,
            yaw_mode=args.yaw_mode,
            heading_lookahead=args.heading_lookahead,
            corner_interval=args.corner_interval,
            corner_yaw_threshold_deg=args.corner_yaw_threshold_deg,
            corner_expand_distance=args.corner_expand_distance,
        )
        write_waypoints_yaml(
            args.output,
            waypoints,
            frame_id=args.frame_id,
            source_path=args.input,
            raw_count=len(samples),
            filtered_count=len(filtered),
            interval=args.interval,
            path_length=path_length,
            loop=args.loop,
            yaw_mode=args.yaw_mode,
            corner_sampling=corner_sampling,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"read poses: raw={len(samples)}, filtered={len(filtered)}")
    print(f"path length: {path_length:.3f} m")
    if corner_sampling.interval > 0.0:
        print(f"corner interval: {corner_sampling.interval:.3f} m")
        print(f"corner zones: {format_distance_zones(corner_sampling.zones)}")
    print(f"generated waypoints: {len(waypoints)}")
    print(f"output: {os.path.abspath(args.output)}")
    if waypoints:
        print(
            "first waypoint: "
            f"x={waypoints[0].x:.3f}, y={waypoints[0].y:.3f}, "
            f"yaw={math.degrees(waypoints[0].yaw):.1f} deg"
        )
        print(
            "last waypoint: "
            f"x={waypoints[-1].x:.3f}, y={waypoints[-1].y:.3f}, "
            f"yaw={math.degrees(waypoints[-1].yaw):.1f} deg"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

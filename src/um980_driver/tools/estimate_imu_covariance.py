#!/usr/bin/env python3
"""Estimate IMU covariance from a static rosbag."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import rosbag


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def vector_mean(samples: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return (
        statistics.fmean(sample[0] for sample in samples),
        statistics.fmean(sample[1] for sample in samples),
        statistics.fmean(sample[2] for sample in samples),
    )


def covariance_matrix(samples: list[tuple[float, float, float]]) -> list[list[float]]:
    if not samples:
        raise ValueError("no samples available")
    mean_x, mean_y, mean_z = vector_mean(samples)
    matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
    for sample_x, sample_y, sample_z in samples:
        deltas = (
            sample_x - mean_x,
            sample_y - mean_y,
            sample_z - mean_z,
        )
        for row in range(3):
            for col in range(3):
                matrix[row][col] += deltas[row] * deltas[col]
    sample_count = float(len(samples))
    for row in range(3):
        for col in range(3):
            matrix[row][col] /= sample_count
    return matrix


def flatten_row_major(matrix: list[list[float]]) -> list[float]:
    return [matrix[row][col] for row in range(3) for col in range(3)]


def stddev_from_covariance(matrix: list[list[float]]) -> tuple[float, float, float]:
    return (
        math.sqrt(max(0.0, matrix[0][0])),
        math.sqrt(max(0.0, matrix[1][1])),
        math.sqrt(max(0.0, matrix[2][2])),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate IMU covariance from a static rosbag."
    )
    parser.add_argument("bag", help="Path to the rosbag file")
    parser.add_argument("--imu-topic", default="/IMU_data")
    parser.add_argument(
        "--start-offset",
        type=float,
        default=0.0,
        help="Seconds after bag start to begin using samples",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional analysis duration in seconds; 0 uses the rest of the bag",
    )
    parser.add_argument(
        "--estimate-orientation",
        action="store_true",
        help="Estimate orientation covariance from quaternion jitter instead of disabling it",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        print(f"bag file not found: {bag_path}", file=sys.stderr)
        return 1

    imu_times: list[float] = []
    angular_velocity_samples: list[tuple[float, float, float]] = []
    linear_acceleration_samples: list[tuple[float, float, float]] = []
    orientation_samples: list[tuple[float, float, float]] = []

    with rosbag.Bag(bag_path, "r") as bag:
        bag_start = bag.get_start_time()
        bag_end = bag.get_end_time()
        window_start = bag_start + max(0.0, args.start_offset)
        window_end = bag_end if args.duration <= 0.0 else min(
            bag_end, window_start + args.duration
        )

        for _, msg, _ in bag.read_messages(topics=[args.imu_topic]):
            stamp = msg.header.stamp.to_sec()
            if stamp <= 0.0:
                stamp = bag_start
            if stamp < window_start or stamp > window_end:
                continue

            imu_times.append(stamp)
            angular_velocity_samples.append(
                (
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                )
            )
            linear_acceleration_samples.append(
                (
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                )
            )
            orientation_samples.append(
                (
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                )
            )

    if len(imu_times) < 10:
        print("not enough IMU samples in the selected window", file=sys.stderr)
        return 1

    duration = imu_times[-1] - imu_times[0] if len(imu_times) > 1 else 0.0
    rate = (len(imu_times) - 1) / duration if duration > 0.0 else 0.0

    angular_mean = vector_mean(angular_velocity_samples)
    angular_cov = covariance_matrix(angular_velocity_samples)
    angular_std = stddev_from_covariance(angular_cov)

    linear_mean = vector_mean(linear_acceleration_samples)
    linear_cov = covariance_matrix(linear_acceleration_samples)
    linear_std = stddev_from_covariance(linear_cov)

    if args.estimate_orientation:
        orientation_cov = covariance_matrix(orientation_samples)
    else:
        orientation_cov = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    orientation_flat = flatten_row_major(orientation_cov)
    angular_flat = flatten_row_major(angular_cov)
    linear_flat = flatten_row_major(linear_cov)

    print(f"Bag: {bag_path}")
    print(f"IMU topic: {args.imu_topic}")
    print(
        "Window: "
        f"start_offset={fmt(args.start_offset, 3)} s "
        f"duration={'full' if args.duration <= 0.0 else fmt(args.duration, 3) + ' s'}"
    )
    print(f"Samples: {len(imu_times)}")
    print(f"Rate: {fmt(rate, 3)} Hz")

    print("\nAngular velocity")
    print(
        "mean: "
        f"wx={fmt(angular_mean[0], 6)} "
        f"wy={fmt(angular_mean[1], 6)} "
        f"wz={fmt(angular_mean[2], 6)} rad/s"
    )
    print(
        "std:  "
        f"wx={fmt(angular_std[0], 6)} "
        f"wy={fmt(angular_std[1], 6)} "
        f"wz={fmt(angular_std[2], 6)} rad/s"
    )
    print(f"covariance: {angular_flat}")

    print("\nLinear acceleration")
    print(
        "mean: "
        f"ax={fmt(linear_mean[0], 6)} "
        f"ay={fmt(linear_mean[1], 6)} "
        f"az={fmt(linear_mean[2], 6)} m/s^2"
    )
    print(
        "std:  "
        f"ax={fmt(linear_std[0], 6)} "
        f"ay={fmt(linear_std[1], 6)} "
        f"az={fmt(linear_std[2], 6)} m/s^2"
    )
    print(f"covariance: {linear_flat}")

    print("\nSuggested IMU parameters")
    print(f"orientation_covariance: {orientation_flat}")
    print(f"angular_velocity_covariance: {angular_flat}")
    print(f"linear_acceleration_covariance: {linear_flat}")

    print("\nYAML fragments")
    print(f"orientation_covariance: {orientation_flat}")
    print(f"angular_velocity_covariance: {angular_flat}")
    print(f"linear_acceleration_covariance: {linear_flat}")

    if not args.estimate_orientation:
        print("\nNotes")
        print("- orientation_covariance is left disabled by default.")
        print("- Use a static bag for reliable wx/wy/wz and acceleration noise estimates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

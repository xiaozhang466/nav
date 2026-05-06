#!/usr/bin/env python3
"""Stitch FAST-LOCALIZATION style frame PCDs into one global map PCD.

Input layout:
  map_dir/
    pcd/
      0.pcd
      1.pcd
      ...
    pose.json

pose.json format:
  x y z qw qx qy qz
"""

import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit(
        "open3d is required. Install it with: python3 -m pip install open3d"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stitch FAST-LOCALIZATION pcd/ + pose.json into global_map.pcd."
    )
    parser.add_argument(
        "--map-dir",
        required=True,
        help="Directory containing pcd/ and pose.json.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output PCD path. Default: <map-dir>/global_map.pcd.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.10,
        help="Voxel downsample size in meters. Use 0 to disable. Default: 0.10.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame. Default: 1.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First pose/frame index to use. Default: 0.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to stitch. 0 means all.",
    )
    parser.add_argument(
        "--periodic-downsample-frames",
        type=int,
        default=200,
        help="Downsample accumulated cloud every N stitched frames when voxel-size > 0. Default: 200.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N stitched frames. Default: 100.",
    )
    return parser.parse_args()


def load_poses(pose_path: Path):
    poses = []
    with pose_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 7:
                raise ValueError(
                    f"{pose_path}:{line_no} expected 7 fields: x y z qw qx qy qz"
                )
            x, y, z, qw, qx, qy, qz = map(float, parts)
            q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if q_norm <= 1e-12:
                raise ValueError(f"{pose_path}:{line_no} quaternion norm is zero")
            poses.append(
                (
                    np.array([x, y, z], dtype=np.float64),
                    np.array([qw, qx, qy, qz], dtype=np.float64) / q_norm,
                )
            )
    if not poses:
        raise ValueError(f"No valid poses found in {pose_path}")
    return poses


def find_pcd_path(pcd_dir: Path, index: int):
    candidates = (
        pcd_dir / f"{index}.pcd",
        pcd_dir / f"{index:06d}.pcd",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def transform_from_pose(translation, quaternion_wxyz):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
        quaternion_wxyz
    )
    transform[:3, 3] = translation
    return transform


def downsample_if_needed(cloud, voxel_size):
    if voxel_size <= 0.0 or cloud.is_empty():
        return cloud
    return cloud.voxel_down_sample(voxel_size)


def main():
    args = parse_args()
    map_dir = Path(os.path.expanduser(args.map_dir)).resolve()
    pcd_dir = map_dir / "pcd"
    pose_path = map_dir / "pose.json"
    output_path = (
        Path(os.path.expanduser(args.output)).resolve()
        if args.output
        else map_dir / "global_map.pcd"
    )

    if args.stride <= 0:
        raise SystemExit("--stride must be >= 1")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be >= 0")
    if args.voxel_size < 0.0:
        raise SystemExit("--voxel-size must be >= 0")
    if not pcd_dir.is_dir():
        raise SystemExit(f"pcd directory not found: {pcd_dir}")
    if not pose_path.is_file():
        raise SystemExit(f"pose.json not found: {pose_path}")

    poses = load_poses(pose_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_cloud = o3d.geometry.PointCloud()
    used = 0
    missing = 0
    failed = 0
    stop_index = len(poses)
    if args.max_frames > 0:
        stop_index = min(stop_index, args.start_index + args.max_frames * args.stride)

    for index in range(args.start_index, stop_index, args.stride):
        pcd_path = find_pcd_path(pcd_dir, index)
        if pcd_path is None:
            missing += 1
            continue

        cloud = o3d.io.read_point_cloud(str(pcd_path))
        if cloud.is_empty():
            failed += 1
            print(f"warning: empty or unreadable PCD: {pcd_path}", file=sys.stderr)
            continue

        translation, quaternion = poses[index]
        cloud.transform(transform_from_pose(translation, quaternion))
        global_cloud += cloud
        used += 1

        if (
            args.voxel_size > 0.0
            and args.periodic_downsample_frames > 0
            and used % args.periodic_downsample_frames == 0
        ):
            global_cloud = downsample_if_needed(global_cloud, args.voxel_size)

        if args.progress_every > 0 and used % args.progress_every == 0:
            print(
                f"stitched {used} frames, current points={len(global_cloud.points)}"
            )

    if used == 0:
        raise SystemExit(
            f"No frames stitched. Check pose/pcd naming under {map_dir}"
        )

    global_cloud = downsample_if_needed(global_cloud, args.voxel_size)
    ok = o3d.io.write_point_cloud(str(output_path), global_cloud)
    if not ok:
        raise SystemExit(f"failed to write output PCD: {output_path}")

    print("stitch complete")
    print(f"  map_dir: {map_dir}")
    print(f"  output: {output_path}")
    print(f"  poses: {len(poses)}")
    print(f"  used_frames: {used}")
    print(f"  missing_pcd: {missing}")
    print(f"  failed_pcd: {failed}")
    print(f"  voxel_size: {args.voxel_size}")
    print(f"  output_points: {len(global_cloud.points)}")


if __name__ == "__main__":
    main()

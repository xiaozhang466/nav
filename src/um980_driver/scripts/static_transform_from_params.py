#!/usr/bin/env python3
"""Publish one static TF from ROS parameters."""

from __future__ import annotations

import math

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def nested_param(group_name: str, key: str, default: float) -> float:
    group = rospy.get_param(f"~{group_name}", {})
    if isinstance(group, dict) and key in group:
        return float(group[key])
    return float(rospy.get_param(f"~{key}", default))


def main() -> None:
    rospy.init_node("static_transform_from_params")

    parent_frame = rospy.get_param("~parent_frame", "map")
    child_frame = rospy.get_param("~child_frame", "rtk_map")
    x = nested_param("translation", "x", 0.0)
    y = nested_param("translation", "y", 0.0)
    z = nested_param("translation", "z", 0.0)
    yaw = nested_param("rotation", "yaw", 0.0)
    pitch = nested_param("rotation", "pitch", 0.0)
    roll = nested_param("rotation", "roll", 0.0)

    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.translation.z = z
    qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(transform)
    rospy.loginfo(
        "Publishing static TF %s -> %s: x=%.6f y=%.6f z=%.6f roll=%.6f pitch=%.6f yaw=%.6f",
        parent_frame,
        child_frame,
        x,
        y,
        z,
        roll,
        pitch,
        yaw,
    )
    rospy.spin()


if __name__ == "__main__":
    main()

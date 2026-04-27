#!/usr/bin/env python3
"""Republish IMU orientation after converting north-referenced yaw to ROS ENU yaw."""

from __future__ import annotations

import math

import rospy
from sensor_msgs.msg import Imu


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class ImuHeadingCorrection:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/IMU_data")
        self.output_topic = rospy.get_param("~output_topic", "/IMU_data_enu")
        self.frame_id_override = rospy.get_param("~frame_id", "")
        self.north_to_east_deg = float(rospy.get_param("~north_to_east_deg", 90.0))
        self.yaw_bias_deg = float(rospy.get_param("~yaw_bias_deg", 0.0))
        self.total_yaw_offset_rad = math.radians(self.north_to_east_deg + self.yaw_bias_deg)

        self.pub = rospy.Publisher(self.output_topic, Imu, queue_size=20)
        self.sub = rospy.Subscriber(self.input_topic, Imu, self.callback, queue_size=20)

        rospy.loginfo(
            "imu_heading_correction: %s -> %s, north_to_east_deg=%.3f, yaw_bias_deg=%.3f",
            self.input_topic,
            self.output_topic,
            self.north_to_east_deg,
            self.yaw_bias_deg,
        )

    def callback(self, msg: Imu) -> None:
        roll, pitch, yaw = quaternion_to_rpy(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        corrected_yaw = wrap_angle(yaw + self.total_yaw_offset_rad)
        qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, corrected_yaw)

        corrected = Imu()
        corrected.header = msg.header
        if self.frame_id_override:
            corrected.header.frame_id = self.frame_id_override

        corrected.orientation.x = qx
        corrected.orientation.y = qy
        corrected.orientation.z = qz
        corrected.orientation.w = qw
        corrected.orientation_covariance = msg.orientation_covariance
        corrected.angular_velocity = msg.angular_velocity
        corrected.angular_velocity_covariance = msg.angular_velocity_covariance
        corrected.linear_acceleration = msg.linear_acceleration
        corrected.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(corrected)


def main() -> None:
    rospy.init_node("imu_heading_correction", anonymous=False)
    ImuHeadingCorrection()
    rospy.spin()


if __name__ == "__main__":
    main()

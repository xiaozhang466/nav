#!/usr/bin/env python3
import math
import threading

import rospy
from geometry_msgs.msg import Twist


def clamp(value, limit):
    if limit <= 0.0:
        return value
    return max(-limit, min(limit, value))


def approach(current, target, accel_limit, decel_limit, dt, epsilon):
    if abs(target) < epsilon:
        target = 0.0
    if abs(current) < epsilon:
        current = 0.0

    delta = target - current
    if abs(delta) < epsilon:
        return target

    # Use the deceleration limit when reducing speed or changing direction.
    reducing_speed = abs(target) < abs(current) or (current * target) < 0.0
    limit = decel_limit if reducing_speed else accel_limit
    max_delta = max(0.0, limit) * dt
    if max_delta <= 0.0:
        return target

    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


class CmdVelSmoother:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/cmd_vel_nav")
        self.output_topic = rospy.get_param("~output_topic", "/cmd_vel")
        self.rate = rospy.get_param("~rate", 20.0)
        self.input_timeout = rospy.get_param("~input_timeout", 0.4)
        self.zero_epsilon = rospy.get_param("~zero_epsilon", 1e-3)

        self.linear_accel = rospy.get_param("~linear_accel", 0.25)
        self.linear_decel = rospy.get_param("~linear_decel", 0.35)
        self.angular_accel = rospy.get_param("~angular_accel", 0.45)
        self.angular_decel = rospy.get_param("~angular_decel", 0.60)
        self.max_linear_x = rospy.get_param("~max_linear_x", 0.25)
        self.max_angular_z = rospy.get_param("~max_angular_z", 0.55)

        self._lock = threading.Lock()
        self._target = Twist()
        self._current = Twist()
        self._last_input_time = rospy.Time(0)

        self._pub = rospy.Publisher(self.output_topic, Twist, queue_size=1)
        self._sub = rospy.Subscriber(self.input_topic, Twist, self._cmd_cb, queue_size=1)

    def _cmd_cb(self, msg):
        target = Twist()
        target.linear.x = clamp(msg.linear.x, self.max_linear_x)
        target.angular.z = clamp(msg.angular.z, self.max_angular_z)

        with self._lock:
            self._target = target
            self._last_input_time = rospy.Time.now()

    def _target_or_zero(self):
        with self._lock:
            if self._last_input_time == rospy.Time(0):
                return Twist()
            if (rospy.Time.now() - self._last_input_time).to_sec() > self.input_timeout:
                return Twist()
            return self._target

    def spin(self):
        rate = rospy.Rate(self.rate)
        last_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = max(0.0, (now - last_time).to_sec())
            last_time = now

            target = self._target_or_zero()
            self._current.linear.x = approach(
                self._current.linear.x,
                target.linear.x,
                self.linear_accel,
                self.linear_decel,
                dt,
                self.zero_epsilon,
            )
            self._current.angular.z = approach(
                self._current.angular.z,
                target.angular.z,
                self.angular_accel,
                self.angular_decel,
                dt,
                self.zero_epsilon,
            )
            self._pub.publish(self._current)
            rate.sleep()

        self._pub.publish(Twist())


if __name__ == "__main__":
    rospy.init_node("cmd_vel_smoother")
    CmdVelSmoother().spin()

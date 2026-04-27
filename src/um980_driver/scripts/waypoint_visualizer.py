#!/usr/bin/env python3
"""Publish RViz visualization for patrol waypoints."""

import math
import os
import sys
from typing import Dict, List

import rospy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray


def yaw_to_quaternion(yaw: float):
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def color(r: float, g: float, b: float, a: float = 1.0):
    return (r, g, b, a)


def set_marker_color(marker: Marker, rgba):
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.color.a = rgba[3]


def load_waypoints(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("waypoint file must be a YAML mapping")

    frame_id = str(data.get("frame_id", "map"))
    raw_waypoints = data.get("waypoints", [])
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise ValueError("waypoint file must contain a non-empty 'waypoints' list")

    waypoints: List[Dict[str, float]] = []
    for index, item in enumerate(raw_waypoints):
        if not isinstance(item, dict):
            raise ValueError(f"waypoint {index} is not a mapping")
        try:
            waypoint_id = int(item.get("id", index))
            x = float(item["x"])
            y = float(item["y"])
            z = float(item.get("z", 0.0))
            yaw = float(item.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"waypoint {index} is missing numeric x/y/yaw fields") from exc
        waypoints.append({"id": waypoint_id, "x": x, "y": y, "z": z, "yaw": yaw})

    return frame_id, waypoints


class WaypointVisualizer:
    def __init__(self):
        self.waypoints_file = rospy.get_param("~waypoints_file")
        self.frame_override = rospy.get_param("~frame_id", "")
        self.path_topic = rospy.get_param("~path_topic", "/waypoints/path")
        self.marker_topic = rospy.get_param("~marker_topic", "/waypoints/markers")
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))
        self.z_offset = float(rospy.get_param("~z_offset", 0.08))
        self.point_scale = float(rospy.get_param("~point_scale", 0.28))
        self.arrow_length = float(rospy.get_param("~arrow_length", 0.9))
        self.arrow_shaft = float(rospy.get_param("~arrow_shaft", 0.08))
        self.arrow_head = float(rospy.get_param("~arrow_head", 0.18))
        self.text_scale = float(rospy.get_param("~text_scale", 0.5))

        if not self.waypoints_file:
            raise ValueError("~waypoints_file is required")

        self.frame_id, self.waypoints = load_waypoints(self.waypoints_file)
        if self.frame_override:
            self.frame_id = self.frame_override

        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1, latch=True)

        self.publish()
        period = 1.0 / max(self.publish_rate, 0.1)
        self.timer = rospy.Timer(rospy.Duration(period), lambda _event: self.publish())

        rospy.loginfo(
            "Loaded %d waypoints from %s, publishing %s and %s in frame %s",
            len(self.waypoints),
            os.path.abspath(self.waypoints_file),
            self.path_topic,
            self.marker_topic,
            self.frame_id,
        )

    def make_path(self, stamp: rospy.Time) -> Path:
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id
        for waypoint in self.waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = waypoint["x"]
            pose.pose.position.y = waypoint["y"]
            pose.pose.position.z = waypoint["z"] + self.z_offset
            qx, qy, qz, qw = yaw_to_quaternion(waypoint["yaw"])
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        return path

    def make_marker_header(self, marker: Marker, stamp: rospy.Time):
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.0)

    def make_markers(self, stamp: rospy.Time) -> MarkerArray:
        markers = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.header.stamp = stamp
        clear.header.frame_id = self.frame_id
        markers.markers.append(clear)

        line = Marker()
        self.make_marker_header(line, stamp)
        line.ns = "waypoint_route"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.scale.x = 0.08
        set_marker_color(line, color(0.0, 0.85, 1.0, 0.85))
        for waypoint in self.waypoints:
            line.points.append(
                Point(
                    x=waypoint["x"],
                    y=waypoint["y"],
                    z=waypoint["z"] + self.z_offset,
                )
            )
        markers.markers.append(line)

        for index, waypoint in enumerate(self.waypoints):
            point_color = color(0.0, 0.75, 1.0, 1.0)
            if index == 0:
                point_color = color(0.0, 1.0, 0.2, 1.0)
            elif index == len(self.waypoints) - 1:
                point_color = color(1.0, 0.45, 0.0, 1.0)

            sphere = Marker()
            self.make_marker_header(sphere, stamp)
            sphere.ns = "waypoint_points"
            sphere.id = 1000 + index
            sphere.type = Marker.SPHERE
            sphere.pose.position.x = waypoint["x"]
            sphere.pose.position.y = waypoint["y"]
            sphere.pose.position.z = waypoint["z"] + self.z_offset
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = self.point_scale
            sphere.scale.y = self.point_scale
            sphere.scale.z = self.point_scale
            set_marker_color(sphere, point_color)
            markers.markers.append(sphere)

            arrow = Marker()
            self.make_marker_header(arrow, stamp)
            arrow.ns = "waypoint_heading"
            arrow.id = 2000 + index
            arrow.type = Marker.ARROW
            arrow.scale.x = self.arrow_shaft
            arrow.scale.y = self.arrow_head
            arrow.scale.z = self.arrow_head
            set_marker_color(arrow, color(1.0, 0.9, 0.05, 0.95))
            start = Point(x=waypoint["x"], y=waypoint["y"], z=waypoint["z"] + self.z_offset + 0.05)
            end = Point(
                x=waypoint["x"] + self.arrow_length * math.cos(waypoint["yaw"]),
                y=waypoint["y"] + self.arrow_length * math.sin(waypoint["yaw"]),
                z=waypoint["z"] + self.z_offset + 0.05,
            )
            arrow.points = [start, end]
            markers.markers.append(arrow)

            text = Marker()
            self.make_marker_header(text, stamp)
            text.ns = "waypoint_labels"
            text.id = 3000 + index
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = waypoint["x"]
            text.pose.position.y = waypoint["y"]
            text.pose.position.z = waypoint["z"] + self.z_offset + 0.55
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_scale
            text.text = str(waypoint["id"])
            set_marker_color(text, color(1.0, 1.0, 1.0, 1.0))
            markers.markers.append(text)

        return markers

    def publish(self):
        stamp = rospy.Time.now()
        self.path_pub.publish(self.make_path(stamp))
        self.marker_pub.publish(self.make_markers(stamp))


def main() -> int:
    rospy.init_node("waypoint_visualizer")
    try:
        WaypointVisualizer()
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logerr("Failed to start waypoint visualizer: %s", exc)
        return 1
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())

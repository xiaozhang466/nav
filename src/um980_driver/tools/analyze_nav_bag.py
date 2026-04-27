#!/usr/bin/env python3
import argparse
import bisect
import collections
import math
import os
import statistics
import sys

import rosbag


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(delta):
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def mean_and_std(values):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, stddev


def format_float(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_section(title):
    print(f"\n== {title} ==")


def bag_rate(times):
    if len(times) < 2:
        return None
    duration = times[-1] - times[0]
    if duration <= 0.0:
        return None
    return (len(times) - 1) / duration


def extract_topic_counts(bag):
    info = bag.get_type_and_topic_info()
    counts = {}
    for topic, topic_info in info.topics.items():
        counts[topic] = topic_info.message_count
    return counts


def analyze_bag(args):
    bag_path = os.path.abspath(args.bag)
    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"bag file not found: {bag_path}")

    odom = {
        "times": [],
        "x": [],
        "y": [],
        "yaw": [],
        "vx": [],
        "vy": [],
        "wz": [],
        "pose_cov": None,
        "twist_cov": None,
    }
    imu = {
        "times": [],
        "frame_ids": set(),
        "wx": [],
        "wy": [],
        "wz": [],
        "ax": [],
        "ay": [],
        "az": [],
        "orientation_cov": None,
        "angular_cov": None,
        "linear_cov": None,
    }
    fix = {
        "times": [],
        "lat": [],
        "lon": [],
        "alt": [],
        "status": [],
        "service": [],
    }
    fix_type_counter = collections.Counter()
    position_type_counter = collections.Counter()

    with rosbag.Bag(bag_path, "r") as bag:
        bag_counts = extract_topic_counts(bag)
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()

        for topic, msg, t in bag.read_messages():
            stamp = t.to_sec()

            if topic == args.odom_topic:
                odom["times"].append(stamp)
                odom["x"].append(msg.pose.pose.position.x)
                odom["y"].append(msg.pose.pose.position.y)
                odom["yaw"].append(yaw_from_quaternion(msg.pose.pose.orientation))
                odom["vx"].append(msg.twist.twist.linear.x)
                odom["vy"].append(msg.twist.twist.linear.y)
                odom["wz"].append(msg.twist.twist.angular.z)
                if odom["pose_cov"] is None:
                    odom["pose_cov"] = list(msg.pose.covariance)
                if odom["twist_cov"] is None:
                    odom["twist_cov"] = list(msg.twist.covariance)

            elif topic == args.imu_topic:
                imu["times"].append(stamp)
                imu["frame_ids"].add(msg.header.frame_id)
                imu["wx"].append(msg.angular_velocity.x)
                imu["wy"].append(msg.angular_velocity.y)
                imu["wz"].append(msg.angular_velocity.z)
                imu["ax"].append(msg.linear_acceleration.x)
                imu["ay"].append(msg.linear_acceleration.y)
                imu["az"].append(msg.linear_acceleration.z)
                if imu["orientation_cov"] is None:
                    imu["orientation_cov"] = list(msg.orientation_covariance)
                    imu["angular_cov"] = list(msg.angular_velocity_covariance)
                    imu["linear_cov"] = list(msg.linear_acceleration_covariance)

            elif topic == args.fix_topic:
                fix["times"].append(stamp)
                fix["lat"].append(msg.latitude)
                fix["lon"].append(msg.longitude)
                fix["alt"].append(msg.altitude)
                fix["status"].append(msg.status.status)
                fix["service"].append(msg.status.service)

            elif topic == args.fix_type_topic:
                data = msg.data.strip()
                if data:
                    fix_type_counter[data] += 1

            elif topic == args.position_type_topic:
                data = msg.data.strip()
                if data:
                    position_type_counter[data] += 1

    print(f"Bag: {bag_path}")
    print(f"Duration: {format_float(end_time - start_time, 2)} s")
    print(f"Time span: {format_float(start_time, 3)} -> {format_float(end_time, 3)}")

    print_section("Topics")
    interesting_topics = [
        args.odom_topic,
        args.imu_topic,
        args.fix_topic,
        args.fix_type_topic,
        args.position_type_topic,
        "/motion_state",
        "/system_state",
        "/cmd_vel",
        "/tf",
        "/tf_static",
    ]
    for topic in interesting_topics:
        if topic in bag_counts:
            print(f"{topic}: {bag_counts[topic]} msgs")

    if odom["times"]:
        print_section("Odometry")
        dx = odom["x"][-1] - odom["x"][0]
        dy = odom["y"][-1] - odom["y"][0]
        displacement = math.hypot(dx, dy)
        path_length = 0.0
        yaw_delta = 0.0
        for i in range(1, len(odom["times"])):
            path_length += math.hypot(
                odom["x"][i] - odom["x"][i - 1], odom["y"][i] - odom["y"][i - 1]
            )
            yaw_delta += wrap_angle(odom["yaw"][i] - odom["yaw"][i - 1])

        print(f"Topic: {args.odom_topic}")
        print(f"Samples: {len(odom['times'])}")
        print(f"Rate: {format_float(bag_rate(odom['times']), 2)} Hz")
        print(
            "Start pose: "
            f"x={format_float(odom['x'][0], 3)} y={format_float(odom['y'][0], 3)} "
            f"yaw={format_float(math.degrees(odom['yaw'][0]), 2)} deg"
        )
        print(
            "End pose: "
            f"x={format_float(odom['x'][-1], 3)} y={format_float(odom['y'][-1], 3)} "
            f"yaw={format_float(math.degrees(odom['yaw'][-1]), 2)} deg"
        )
        print(
            f"Net displacement: {format_float(displacement, 3)} m "
            f"(dx={format_float(dx, 3)}, dy={format_float(dy, 3)})"
        )
        print(f"Integrated path length: {format_float(path_length, 3)} m")
        print(f"Integrated yaw change: {format_float(math.degrees(yaw_delta), 2)} deg")
        print(
            "Max |twist|: "
            f"|vx|={format_float(max(abs(v) for v in odom['vx']), 3)} m/s "
            f"|vy|={format_float(max(abs(v) for v in odom['vy']), 3)} m/s "
            f"|wz|={format_float(max(abs(v) for v in odom['wz']), 3)} rad/s"
        )

        stationary_flags = []
        for vx, vy, wz in zip(odom["vx"], odom["vy"], odom["wz"]):
            stationary_flags.append(
                abs(vx) < args.stationary_linear_threshold
                and abs(vy) < args.stationary_linear_threshold
                and abs(wz) < args.stationary_angular_threshold
            )
        stationary_ratio = sum(1 for flag in stationary_flags if flag) / len(stationary_flags)
        print(f"Stationary sample ratio: {format_float(stationary_ratio * 100.0, 1)} %")

        if odom["pose_cov"]:
            pose_diag = [odom["pose_cov"][i] for i in (0, 7, 14, 21, 28, 35)]
            twist_diag = [odom["twist_cov"][i] for i in (0, 7, 14, 21, 28, 35)]
            print(f"Pose covariance diag: {pose_diag}")
            print(f"Twist covariance diag: {twist_diag}")
    else:
        stationary_flags = []
        print_section("Odometry")
        print(f"No messages found on {args.odom_topic}")

    if imu["times"]:
        print_section("IMU")
        print(f"Topic: {args.imu_topic}")
        print(f"Samples: {len(imu['times'])}")
        print(f"Rate: {format_float(bag_rate(imu['times']), 2)} Hz")
        print(f"frame_id values: {sorted(imu['frame_ids'])}")

        for label, values in (
            ("angular_velocity.x", imu["wx"]),
            ("angular_velocity.y", imu["wy"]),
            ("angular_velocity.z", imu["wz"]),
            ("linear_acceleration.x", imu["ax"]),
            ("linear_acceleration.y", imu["ay"]),
            ("linear_acceleration.z", imu["az"]),
        ):
            mean, stddev = mean_and_std(values)
            print(f"{label}: mean={format_float(mean, 4)} std={format_float(stddev, 4)}")

        if imu["orientation_cov"] is not None:
            print(f"orientation_covariance: {imu['orientation_cov']}")
            print(f"angular_velocity_covariance: {imu['angular_cov']}")
            print(f"linear_acceleration_covariance: {imu['linear_cov']}")

        if odom["times"]:
            odom_times = odom["times"]
            imu_static_indices = []
            for i, stamp in enumerate(imu["times"]):
                odom_idx = bisect.bisect_right(odom_times, stamp) - 1
                if odom_idx >= 0 and stationary_flags[odom_idx]:
                    imu_static_indices.append(i)

            if imu_static_indices:
                print("IMU during odom-stationary segments:")
                for label, values in (
                    ("angular_velocity.x", imu["wx"]),
                    ("angular_velocity.y", imu["wy"]),
                    ("angular_velocity.z", imu["wz"]),
                    ("linear_acceleration.z", imu["az"]),
                ):
                    subset = [values[i] for i in imu_static_indices]
                    mean, stddev = mean_and_std(subset)
                    print(
                        f"{label}: mean={format_float(mean, 4)} "
                        f"std={format_float(stddev, 4)}"
                    )
    else:
        print_section("IMU")
        print(f"No messages found on {args.imu_topic}")

    if fix["times"]:
        print_section("RTK")
        print(f"Topic: {args.fix_topic}")
        print(f"Samples: {len(fix['times'])}")
        print(f"Rate: {format_float(bag_rate(fix['times']), 2)} Hz")
        print(
            "Latitude span: "
            f"{format_float(min(fix['lat']), 8)} -> {format_float(max(fix['lat']), 8)}"
        )
        print(
            "Longitude span: "
            f"{format_float(min(fix['lon']), 8)} -> {format_float(max(fix['lon']), 8)}"
        )
        print(
            "Altitude span: "
            f"{format_float(min(fix['alt']), 3)} -> {format_float(max(fix['alt']), 3)} m"
        )
        print(f"NavSat status values: {sorted(set(fix['status']))}")
        print(f"NavSat service values: {sorted(set(fix['service']))}")
    else:
        print_section("RTK")
        print(f"No messages found on {args.fix_topic}")

    if fix_type_counter:
        print(f"RTK fix types: {dict(fix_type_counter)}")
    if position_type_counter:
        print(f"RTK position types: {dict(position_type_counter)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze a navigation rosbag containing chassis odom, IMU, and RTK topics."
    )
    parser.add_argument("bag", help="Path to the rosbag file")
    parser.add_argument("--odom-topic", default="/ranger_base_node/odom")
    parser.add_argument("--imu-topic", default="/IMU_data")
    parser.add_argument("--fix-topic", default="/rtk/fix")
    parser.add_argument("--fix-type-topic", default="/rtk/fix_type")
    parser.add_argument("--position-type-topic", default="/rtk/position_type")
    parser.add_argument("--stationary-linear-threshold", type=float, default=0.03)
    parser.add_argument("--stationary-angular-threshold", type=float, default=0.05)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        analyze_bag(args)
    except Exception as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

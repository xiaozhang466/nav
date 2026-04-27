#!/usr/bin/env python3
"""Analyze local EKF fusion bag: compare wheel odom vs filtered odom vs IMU."""

import math
import sys
import rosbag
import json

def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def main():
    bag_path = sys.argv[1] if len(sys.argv) > 1 else "/home/sigu/RTK/44/bags/local_ekf_test_20260405_164554.bag"

    odom_t, odom_x, odom_y, odom_yaw, odom_vx, odom_vy, odom_wz = [], [], [], [], [], [], []
    filt_t, filt_x, filt_y, filt_yaw, filt_vx, filt_vy, filt_wz = [], [], [], [], [], [], []
    imu_t, imu_wx, imu_wy, imu_wz_list, imu_ax, imu_ay, imu_az = [], [], [], [], [], [], []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            stamp = t.to_sec()
            if topic == '/ranger_base_node/odom':
                odom_t.append(stamp)
                odom_x.append(msg.pose.pose.position.x)
                odom_y.append(msg.pose.pose.position.y)
                odom_yaw.append(yaw_from_quat(msg.pose.pose.orientation))
                odom_vx.append(msg.twist.twist.linear.x)
                odom_vy.append(msg.twist.twist.linear.y)
                odom_wz.append(msg.twist.twist.angular.z)
            elif topic == '/odometry/filtered':
                filt_t.append(stamp)
                filt_x.append(msg.pose.pose.position.x)
                filt_y.append(msg.pose.pose.position.y)
                filt_yaw.append(yaw_from_quat(msg.pose.pose.orientation))
                filt_vx.append(msg.twist.twist.linear.x)
                filt_vy.append(msg.twist.twist.linear.y)
                filt_wz.append(msg.twist.twist.angular.z)
            elif topic == '/IMU_data':
                imu_t.append(stamp)
                imu_wx.append(msg.angular_velocity.x)
                imu_wy.append(msg.angular_velocity.y)
                imu_wz_list.append(msg.angular_velocity.z)
                imu_ax.append(msg.linear_acceleration.x)
                imu_ay.append(msg.linear_acceleration.y)
                imu_az.append(msg.linear_acceleration.z)

    t0 = min(odom_t[0], filt_t[0], imu_t[0])
    duration = max(odom_t[-1], filt_t[-1], imu_t[-1]) - t0

    print("=" * 70)
    print("           Local EKF Fusion Analysis Report")
    print("=" * 70)
    print(f"Duration:        {duration:.1f} s")
    print(f"Odom messages:   {len(odom_t)}  ({len(odom_t)/duration:.1f} Hz)")
    print(f"Filter messages: {len(filt_t)}  ({len(filt_t)/duration:.1f} Hz)")
    print(f"IMU messages:    {len(imu_t)}  ({len(imu_t)/duration:.1f} Hz)")

    # === 1. Static segment analysis (first 20s) ===
    print("\n" + "-" * 70)
    print("  [1] Static Segment Analysis (first 20s)")
    print("-" * 70)

    static_imu_wz = [wz for t, wz in zip(imu_t, imu_wz_list) if t - t0 < 20]
    static_imu_az = [az for t, az in zip(imu_t, imu_az) if t - t0 < 20]
    static_odom_vx = [vx for t, vx in zip(odom_t, odom_vx) if t - t0 < 20]
    static_odom_wz = [wz for t, wz in zip(odom_t, odom_wz) if t - t0 < 20]

    if static_imu_wz:
        wz_mean = sum(static_imu_wz) / len(static_imu_wz)
        wz_std = (sum((w - wz_mean)**2 for w in static_imu_wz) / len(static_imu_wz)) ** 0.5
        print(f"  IMU wz bias:     {wz_mean:.6f} rad/s  (std={wz_std:.6f})")
        if abs(wz_mean) > 0.01:
            print(f"  ⚠️  IMU gyro z-bias is significant! May cause yaw drift.")
        else:
            print(f"  ✅  IMU gyro z-bias is small.")

    if static_imu_az:
        az_mean = sum(static_imu_az) / len(static_imu_az)
        print(f"  IMU az mean:     {az_mean:.4f} m/s² (expect ≈9.8 if gravity included)")
        if abs(az_mean) < 1.0:
            print(f"  → IMU outputs gravity-free linear acceleration")
        elif abs(az_mean - 9.8) < 1.0:
            print(f"  → IMU outputs raw acceleration (includes gravity)")
        else:
            print(f"  ⚠️  Unexpected az value, check IMU orientation")

    if static_odom_vx:
        ovx_mean = sum(abs(v) for v in static_odom_vx) / len(static_odom_vx)
        owz_mean = sum(abs(w) for w in static_odom_wz) / len(static_odom_wz)
        print(f"  Odom |vx| mean:  {ovx_mean:.6f} m/s  (should be ≈0)")
        print(f"  Odom |wz| mean:  {owz_mean:.6f} rad/s (should be ≈0)")

    # === 2. Filtered odom static drift ===
    static_filt_x = [x for t, x in zip(filt_t, filt_x) if t - t0 < 20]
    static_filt_y = [y for t, y in zip(filt_t, filt_y) if t - t0 < 20]
    static_filt_yaw = [y for t, y in zip(filt_t, filt_yaw) if t - t0 < 20]
    if len(static_filt_x) > 10:
        dx = static_filt_x[-1] - static_filt_x[0]
        dy = static_filt_y[-1] - static_filt_y[0]
        dyaw = wrap(static_filt_yaw[-1] - static_filt_yaw[0])
        print(f"\n  Filtered drift during static:")
        print(f"    dx={dx:.4f}m  dy={dy:.4f}m  dyaw={math.degrees(dyaw):.2f}°")
        drift = math.hypot(dx, dy)
        if drift > 0.05:
            print(f"  ⚠️  Position drifted {drift:.3f}m while static!")
        else:
            print(f"  ✅  Position stable while static.")

    # === 3. Overall trajectory comparison ===
    print("\n" + "-" * 70)
    print("  [2] Trajectory Summary")
    print("-" * 70)

    # Wheel odom
    odom_path = 0
    for i in range(1, len(odom_x)):
        odom_path += math.hypot(odom_x[i]-odom_x[i-1], odom_y[i]-odom_y[i-1])
    odom_disp = math.hypot(odom_x[-1]-odom_x[0], odom_y[-1]-odom_y[0])
    odom_dyaw = wrap(odom_yaw[-1] - odom_yaw[0])

    print(f"  Wheel Odom:")
    print(f"    Start: ({odom_x[0]:.3f}, {odom_y[0]:.3f})  yaw={math.degrees(odom_yaw[0]):.1f}°")
    print(f"    End:   ({odom_x[-1]:.3f}, {odom_y[-1]:.3f})  yaw={math.degrees(odom_yaw[-1]):.1f}°")
    print(f"    Path length:   {odom_path:.3f} m")
    print(f"    Displacement:  {odom_disp:.3f} m (should be ≈0 if closed loop)")
    print(f"    Yaw change:    {math.degrees(odom_dyaw):.1f}°")

    # Filtered
    filt_path = 0
    for i in range(1, len(filt_x)):
        filt_path += math.hypot(filt_x[i]-filt_x[i-1], filt_y[i]-filt_y[i-1])
    filt_disp = math.hypot(filt_x[-1]-filt_x[0], filt_y[-1]-filt_y[0])
    filt_dyaw = wrap(filt_yaw[-1] - filt_yaw[0])

    print(f"\n  Filtered Odom:")
    print(f"    Start: ({filt_x[0]:.3f}, {filt_y[0]:.3f})  yaw={math.degrees(filt_yaw[0]):.1f}°")
    print(f"    End:   ({filt_x[-1]:.3f}, {filt_y[-1]:.3f})  yaw={math.degrees(filt_yaw[-1]):.1f}°")
    print(f"    Path length:   {filt_path:.3f} m")
    print(f"    Displacement:  {filt_disp:.3f} m (should be ≈0 if closed loop)")
    print(f"    Yaw change:    {math.degrees(filt_dyaw):.1f}°")

    print(f"\n  Odom vs Filtered endpoint diff:")
    print(f"    dx={abs(odom_x[-1]-filt_x[-1]):.3f}m  dy={abs(odom_y[-1]-filt_y[-1]):.3f}m  dyaw={abs(math.degrees(wrap(odom_yaw[-1]-filt_yaw[-1]))):.1f}°")

    # === 4. Turn analysis ===
    print("\n" + "-" * 70)
    print("  [3] Turn Analysis (yaw rate comparison)")
    print("-" * 70)

    # Find segments where |odom_wz| > 0.1 (turning)
    turn_segments = []
    in_turn = False
    turn_start = 0
    for i, (t, wz) in enumerate(zip(odom_t, odom_wz)):
        if abs(wz) > 0.1 and not in_turn:
            in_turn = True
            turn_start = i
        elif abs(wz) < 0.05 and in_turn:
            in_turn = False
            turn_segments.append((turn_start, i))

    for seg_idx, (si, ei) in enumerate(turn_segments):
        seg_duration = odom_t[ei] - odom_t[si]
        odom_yaw_change = wrap(odom_yaw[ei] - odom_yaw[si])
        filt_yaw_si = min(range(len(filt_t)), key=lambda j: abs(filt_t[j] - odom_t[si]))
        filt_yaw_ei = min(range(len(filt_t)), key=lambda j: abs(filt_t[j] - odom_t[ei]))
        filt_yaw_change = wrap(filt_yaw[filt_yaw_ei] - filt_yaw[filt_yaw_si])

        # IMU integrated yaw change during this segment
        imu_si = min(range(len(imu_t)), key=lambda j: abs(imu_t[j] - odom_t[si]))
        imu_ei = min(range(len(imu_t)), key=lambda j: abs(imu_t[j] - odom_t[ei]))
        imu_yaw_change = 0
        for j in range(imu_si+1, min(imu_ei+1, len(imu_t))):
            dt = imu_t[j] - imu_t[j-1]
            imu_yaw_change += imu_wz_list[j] * dt

        print(f"\n  Turn {seg_idx+1}: t={odom_t[si]-t0:.1f}s ~ {odom_t[ei]-t0:.1f}s  ({seg_duration:.1f}s)")
        print(f"    Wheel odom yaw change:   {math.degrees(odom_yaw_change):+.1f}°")
        print(f"    IMU wz integrated:       {math.degrees(imu_yaw_change):+.1f}°")
        print(f"    Filtered yaw change:     {math.degrees(filt_yaw_change):+.1f}°")
        diff = abs(math.degrees(odom_yaw_change) - math.degrees(imu_yaw_change))
        if diff > 5:
            print(f"    ⚠️  Odom/IMU yaw disagree by {diff:.1f}° — check IMU axis alignment")
        else:
            print(f"    ✅  Odom/IMU yaw consistent (diff={diff:.1f}°)")

    # === 5. Straight-line analysis ===
    print("\n" + "-" * 70)
    print("  [4] Velocity Profile")
    print("-" * 70)
    max_vx = max(abs(v) for v in odom_vx)
    max_wz_odom = max(abs(w) for w in odom_wz)
    max_wz_imu = max(abs(w) for w in imu_wz_list)
    print(f"  Max |odom vx|:   {max_vx:.3f} m/s")
    print(f"  Max |odom wz|:   {max_wz_odom:.3f} rad/s")
    print(f"  Max |IMU wz|:    {max_wz_imu:.3f} rad/s")

    odom_vy_nonzero = [v for v in odom_vy if abs(v) > 0.001]
    filt_vy_nonzero = [v for v in filt_vy if abs(v) > 0.001]
    print(f"  Odom vy non-zero samples: {len(odom_vy_nonzero)} / {len(odom_vy)}")
    print(f"  Filter vy non-zero samples: {len(filt_vy_nonzero)} / {len(filt_vy)}")
    if filt_vy:
        max_filt_vy = max(abs(v) for v in filt_vy)
        print(f"  Max |filtered vy|: {max_filt_vy:.4f} m/s (should be small for diff-drive)")

    # === 6. Diagnostics ===
    print("\n" + "-" * 70)
    print("  [5] EKF Diagnostics")
    print("-" * 70)
    diag_count = 0
    reject_msgs = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/diagnostics']):
            diag_count += 1
            for status in msg.status:
                for kv in status.values:
                    if 'reject' in kv.key.lower() or 'reject' in kv.value.lower():
                        reject_msgs.append(f"  {kv.key}: {kv.value}")

    print(f"  Diagnostic messages: {diag_count}")
    if reject_msgs:
        print(f"  Rejections found:")
        for r in reject_msgs[:10]:
            print(f"    {r}")
    else:
        print(f"  ✅  No data rejections found")

    # === Export for plotting ===
    export = {
        "odom": {"t": [t-t0 for t in odom_t], "x": odom_x, "y": odom_y,
                 "yaw": [math.degrees(y) for y in odom_yaw]},
        "filtered": {"t": [t-t0 for t in filt_t], "x": filt_x, "y": filt_y,
                     "yaw": [math.degrees(y) for y in filt_yaw]},
    }
    with open("/home/sigu/RTK/44/bags/local_ekf_analysis.json", "w") as f:
        json.dump(export, f)
    print(f"\n  Trajectory data exported to bags/local_ekf_analysis.json")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

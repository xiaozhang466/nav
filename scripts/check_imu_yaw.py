#!/usr/bin/env python3
"""Check IMU absolute yaw quality against GPS trajectory."""

import math
import sys
import rosbag

EARTH_R = 6378137.0

def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def latlon_to_xy(lat, lon, lat0, lon0):
    x = (math.radians(lon) - math.radians(lon0)) * math.cos(math.radians(lat0)) * EARTH_R
    y = (math.radians(lat) - math.radians(lat0)) * EARTH_R
    return x, y

def main():
    bag_path = sys.argv[1] if len(sys.argv) > 1 else "/home/sigu/RTK/44/bags/navsat_test_20260405_171612.bag"

    imu_t, imu_yaw = [], []
    odom_t, odom_yaw = [], []
    rtk_t, rtk_lat, rtk_lon = [], [], []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            stamp = t.to_sec()
            if topic == '/IMU_data':
                imu_t.append(stamp)
                imu_yaw.append(yaw_from_quat(msg.orientation))
            elif topic == '/ranger_base_node/odom':
                odom_t.append(stamp)
                odom_yaw.append(yaw_from_quat(msg.pose.pose.orientation))
            elif topic == '/rtk/fix':
                if not math.isnan(msg.latitude):
                    rtk_t.append(stamp)
                    rtk_lat.append(msg.latitude)
                    rtk_lon.append(msg.longitude)

    t0 = min(imu_t[0], odom_t[0], rtk_t[0])

    print("=" * 70)
    print("         IMU Absolute Yaw Analysis")
    print("=" * 70)

    # === 1. IMU yaw during static ===
    print("\n--- [1] IMU Absolute Yaw During Static (first 15s) ---")
    static_yaw = [y for t, y in zip(imu_t, imu_yaw) if t - t0 < 15]
    if static_yaw:
        # Circular mean and std
        sin_sum = sum(math.sin(y) for y in static_yaw)
        cos_sum = sum(math.cos(y) for y in static_yaw)
        mean_yaw = math.atan2(sin_sum / len(static_yaw), cos_sum / len(static_yaw))
        
        diffs = [abs(wrap(y - mean_yaw)) for y in static_yaw]
        std_yaw = (sum(d*d for d in diffs) / len(diffs)) ** 0.5
        
        print(f"  Samples:    {len(static_yaw)}")
        print(f"  Mean yaw:   {math.degrees(mean_yaw):.2f}°")
        print(f"  Std:        {math.degrees(std_yaw):.4f}°")
        
        yaw_drift = wrap(static_yaw[-1] - static_yaw[0])
        print(f"  Drift:      {math.degrees(yaw_drift):.4f}° over 15s")
        
        if math.degrees(std_yaw) < 0.5:
            print(f"  ✅ Yaw is stable during static")
        else:
            print(f"  ⚠️ Yaw is noisy during static")

    # === 2. IMU yaw over full test ===
    print("\n--- [2] IMU Yaw Over Full Test ---")
    first_yaw = imu_yaw[0]
    last_yaw = imu_yaw[-1]
    total_change = wrap(last_yaw - first_yaw)
    print(f"  Start yaw:  {math.degrees(first_yaw):.2f}°")
    print(f"  End yaw:    {math.degrees(last_yaw):.2f}°")
    print(f"  Net change: {math.degrees(total_change):.2f}°")
    
    odom_change = wrap(odom_yaw[-1] - odom_yaw[0])
    print(f"  Odom net change: {math.degrees(odom_change):.2f}°")
    print(f"  Diff (IMU vs odom): {math.degrees(abs(wrap(total_change - odom_change))):.2f}°")

    # === 3. GPS course vs IMU yaw during straight segments ===
    print("\n--- [3] GPS Course vs IMU Yaw (moving straight segments) ---")
    
    if len(rtk_t) < 10:
        print("  Not enough RTK data")
        return
    
    lat0, lon0 = rtk_lat[0], rtk_lon[0]
    rtk_xy = [(t, *latlon_to_xy(lat, lon, lat0, lon0)) for t, lat, lon in zip(rtk_t, rtk_lat, rtk_lon)]
    
    # Compute GPS course (heading from position changes)
    gps_courses = []
    for i in range(2, len(rtk_xy) - 2):
        dt = rtk_xy[i+2][0] - rtk_xy[i-2][0]
        if dt < 0.1:
            continue
        dx = rtk_xy[i+2][1] - rtk_xy[i-2][1]
        dy = rtk_xy[i+2][2] - rtk_xy[i-2][2]
        speed = math.hypot(dx, dy) / dt
        if speed < 0.15:  # Only during movement
            continue
        course = math.atan2(dy, dx)  # ENU: atan2(north, east)
        
        # Find closest IMU yaw
        import bisect
        idx = bisect.bisect_left(imu_t, rtk_xy[i][0])
        if idx >= len(imu_t):
            idx = len(imu_t) - 1
        
        gps_courses.append({
            't': rtk_xy[i][0] - t0,
            'gps_course': course,
            'imu_yaw': imu_yaw[idx],
            'speed': speed,
        })
    
    if gps_courses:
        print(f"  Found {len(gps_courses)} moving samples")
        print(f"\n  {'Time':>6s}  {'GPS course':>11s}  {'IMU yaw':>9s}  {'Diff':>7s}  {'Speed':>6s}")
        print(f"  {'----':>6s}  {'---------':>11s}  {'-------':>9s}  {'----':>7s}  {'-----':>6s}")
        
        offsets = []
        for i, s in enumerate(gps_courses):
            diff = wrap(s['gps_course'] - s['imu_yaw'])
            offsets.append(diff)
            if i % max(1, len(gps_courses)//10) == 0:  # Print ~10 samples
                print(f"  {s['t']:6.1f}s  {math.degrees(s['gps_course']):+9.1f}°  "
                      f"{math.degrees(s['imu_yaw']):+9.1f}°  "
                      f"{math.degrees(diff):+7.1f}°  "
                      f"{s['speed']:5.2f}")
        
        # Circular mean of offset
        sin_sum = sum(math.sin(o) for o in offsets)
        cos_sum = sum(math.cos(o) for o in offsets)
        mean_offset = math.atan2(sin_sum / len(offsets), cos_sum / len(offsets))
        
        diffs_from_mean = [abs(wrap(o - mean_offset)) for o in offsets]
        std_offset = (sum(d*d for d in diffs_from_mean) / len(diffs_from_mean)) ** 0.5
        
        print(f"\n  GPS-IMU yaw offset:")
        print(f"    Mean:  {math.degrees(mean_offset):+.1f}°")
        print(f"    Std:   {math.degrees(std_offset):.1f}°")
        
        if math.degrees(std_offset) < 10:
            print(f"\n  ✅ IMU absolute yaw is consistent with GPS course")
            print(f"     Offset {math.degrees(mean_offset):+.1f}° is likely magnetic declination")
            print(f"     → Can use IMU yaw for navsat_transform!")
            print(f"     → Set use_odometry_yaw: false")
            print(f"     → Set magnetic_declination_radians: {mean_offset:.4f}")
            print(f"     → Set yaw_offset: 0.0")
        else:
            print(f"\n  ❌ IMU absolute yaw is NOT reliable (std={math.degrees(std_offset):.1f}°)")
            print(f"     Do NOT use for navsat_transform")
    else:
        print("  No moving segments found with sufficient speed")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

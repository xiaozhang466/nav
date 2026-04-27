#!/usr/bin/env python3
"""Analyze navsat fusion: compare wheel odom, filtered odom, GPS odom, and raw RTK."""

import math
import sys
import json
import rosbag

def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

EARTH_R = 6378137.0

def latlon_to_xy(lat, lon, lat0, lon0):
    x = (math.radians(lon) - math.radians(lon0)) * math.cos(math.radians(lat0)) * EARTH_R
    y = (math.radians(lat) - math.radians(lat0)) * EARTH_R
    return x, y

def main():
    bag_path = sys.argv[1] if len(sys.argv) > 1 else "/home/sigu/RTK/44/bags/navsat_test_20260405_171612.bag"

    odom_t, odom_x, odom_y, odom_yaw = [], [], [], []
    filt_t, filt_x, filt_y, filt_yaw = [], [], [], []
    gps_odom_t, gps_odom_x, gps_odom_y, gps_odom_yaw = [], [], [], []
    rtk_t, rtk_lat, rtk_lon, rtk_alt, rtk_status, rtk_cov = [], [], [], [], [], []
    fix_types = []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            stamp = t.to_sec()
            if topic == '/ranger_base_node/odom':
                odom_t.append(stamp)
                odom_x.append(msg.pose.pose.position.x)
                odom_y.append(msg.pose.pose.position.y)
                odom_yaw.append(yaw_from_quat(msg.pose.pose.orientation))
            elif topic == '/odometry/filtered':
                filt_t.append(stamp)
                filt_x.append(msg.pose.pose.position.x)
                filt_y.append(msg.pose.pose.position.y)
                filt_yaw.append(yaw_from_quat(msg.pose.pose.orientation))
            elif topic == '/odometry/gps':
                gps_odom_t.append(stamp)
                gps_odom_x.append(msg.pose.pose.position.x)
                gps_odom_y.append(msg.pose.pose.position.y)
                gps_odom_yaw.append(yaw_from_quat(msg.pose.pose.orientation))
            elif topic == '/rtk/fix':
                rtk_t.append(stamp)
                rtk_lat.append(msg.latitude)
                rtk_lon.append(msg.longitude)
                rtk_alt.append(msg.altitude)
                rtk_status.append(msg.status.status)
                rtk_cov.append(msg.position_covariance[0])
            elif topic == '/rtk/fix_type':
                fix_types.append(msg.data.strip())

    t0 = min(odom_t[0], filt_t[0], rtk_t[0])
    duration = max(odom_t[-1], filt_t[-1], rtk_t[-1]) - t0

    print("=" * 70)
    print("         Navsat Transform Analysis Report")
    print("=" * 70)
    print(f"Duration:         {duration:.1f} s")
    print(f"Odom messages:    {len(odom_t)}  ({len(odom_t)/duration:.1f} Hz)")
    print(f"Filtered msgs:    {len(filt_t)}  ({len(filt_t)/duration:.1f} Hz)")
    print(f"GPS odom msgs:    {len(gps_odom_t)}  ({len(gps_odom_t)/duration:.1f} Hz)")
    print(f"RTK fix msgs:     {len(rtk_t)}  ({len(rtk_t)/duration:.1f} Hz)")

    # === 1. RTK quality ===
    print("\n" + "-" * 70)
    print("  [1] RTK Quality")
    print("-" * 70)
    from collections import Counter
    type_counts = Counter(fix_types)
    for ft, cnt in type_counts.most_common():
        pct = cnt / len(fix_types) * 100
        print(f"    {ft:15s}: {cnt:4d} ({pct:.1f}%)")

    status_counts = Counter(rtk_status)
    status_labels = {-1: "NO_FIX", 0: "FIX", 1: "SBAS_FIX", 2: "GBAS_FIX"}
    for st, cnt in sorted(status_counts.items()):
        label = status_labels.get(st, f"UNKNOWN({st})")
        print(f"    NavSat status {st} ({label}): {cnt}")

    # Check for NaN positions
    nan_count = sum(1 for lat in rtk_lat if math.isnan(lat))
    valid_rtk = [(t, lat, lon) for t, lat, lon in zip(rtk_t, rtk_lat, rtk_lon) 
                 if not math.isnan(lat) and not math.isnan(lon)]
    print(f"    Valid positions: {len(valid_rtk)} / {len(rtk_t)}  (NaN: {nan_count})")

    if valid_rtk:
        lat0, lon0 = valid_rtk[0][1], valid_rtk[0][2]
        rtk_xy = [(t, *latlon_to_xy(lat, lon, lat0, lon0)) for t, lat, lon in valid_rtk]
        
        rtk_path = sum(math.hypot(rtk_xy[i][1]-rtk_xy[i-1][1], rtk_xy[i][2]-rtk_xy[i-1][2]) 
                       for i in range(1, len(rtk_xy)))
        rtk_disp = math.hypot(rtk_xy[-1][1]-rtk_xy[0][1], rtk_xy[-1][2]-rtk_xy[0][2])
        
        print(f"\n    RTK local coords (from first valid fix):")
        print(f"    Start: ({rtk_xy[0][1]:.3f}, {rtk_xy[0][2]:.3f}) m")
        print(f"    End:   ({rtk_xy[-1][1]:.3f}, {rtk_xy[-1][2]:.3f}) m")
        print(f"    Path length:  {rtk_path:.3f} m")
        print(f"    Displacement: {rtk_disp:.3f} m")
        
        # RTK position spread
        xs = [p[1] for p in rtk_xy]
        ys = [p[2] for p in rtk_xy]
        print(f"    X range: {min(xs):.3f} ~ {max(xs):.3f} m (span={max(xs)-min(xs):.3f})")
        print(f"    Y range: {min(ys):.3f} ~ {max(ys):.3f} m (span={max(ys)-min(ys):.3f})")

    # === 2. GPS Odom output ===
    print("\n" + "-" * 70)
    print("  [2] Navsat Transform Output (/odometry/gps)")
    print("-" * 70)
    
    if gps_odom_t:
        print(f"    First GPS odom at t={gps_odom_t[0]-t0:.1f}s")
        print(f"    Start: ({gps_odom_x[0]:.3f}, {gps_odom_y[0]:.3f})")
        print(f"    End:   ({gps_odom_x[-1]:.3f}, {gps_odom_y[-1]:.3f})")
        
        gps_path = sum(math.hypot(gps_odom_x[i]-gps_odom_x[i-1], gps_odom_y[i]-gps_odom_y[i-1])
                       for i in range(1, len(gps_odom_x)))
        gps_disp = math.hypot(gps_odom_x[-1]-gps_odom_x[0], gps_odom_y[-1]-gps_odom_y[0])
        print(f"    Path length:  {gps_path:.3f} m")
        print(f"    Displacement: {gps_disp:.3f} m")
        
        gxs = gps_odom_x
        gys = gps_odom_y
        print(f"    X range: {min(gxs):.3f} ~ {max(gxs):.3f} m (span={max(gxs)-min(gxs):.3f})")
        print(f"    Y range: {min(gys):.3f} ~ {max(gys):.3f} m (span={max(gys)-min(gys):.3f})")
    else:
        print("    ⚠️  NO /odometry/gps messages! navsat_transform may not be working.")

    # === 3. Trajectory comparison ===
    print("\n" + "-" * 70)
    print("  [3] Trajectory Comparison")
    print("-" * 70)
    
    odom_path = sum(math.hypot(odom_x[i]-odom_x[i-1], odom_y[i]-odom_y[i-1])
                    for i in range(1, len(odom_x)))
    filt_path = sum(math.hypot(filt_x[i]-filt_x[i-1], filt_y[i]-filt_y[i-1])
                    for i in range(1, len(filt_x)))
    
    print(f"  Wheel Odom:")
    print(f"    Start: ({odom_x[0]:.3f}, {odom_y[0]:.3f}) yaw={math.degrees(odom_yaw[0]):.1f}°")
    print(f"    End:   ({odom_x[-1]:.3f}, {odom_y[-1]:.3f}) yaw={math.degrees(odom_yaw[-1]):.1f}°")
    print(f"    Path:  {odom_path:.3f} m")
    
    print(f"\n  Filtered Odom:")
    print(f"    Start: ({filt_x[0]:.3f}, {filt_y[0]:.3f}) yaw={math.degrees(filt_yaw[0]):.1f}°")
    print(f"    End:   ({filt_x[-1]:.3f}, {filt_y[-1]:.3f}) yaw={math.degrees(filt_yaw[-1]):.1f}°")
    print(f"    Path:  {filt_path:.3f} m")

    # === 4. GPS odom vs odom shape consistency ===
    if gps_odom_t and valid_rtk:
        print("\n" + "-" * 70)
        print("  [4] GPS Odom vs Wheel Odom Alignment Check")
        print("-" * 70)
        
        # Compare path shapes by matching timestamps
        import bisect
        diffs_x, diffs_y, diffs = [], [], []
        for i, gt in enumerate(gps_odom_t):
            oi = bisect.bisect_left(odom_t, gt)
            if oi >= len(odom_t): oi = len(odom_t) - 1
            dx = gps_odom_x[i] - odom_x[oi]
            dy = gps_odom_y[i] - odom_y[oi]
            diffs_x.append(dx)
            diffs_y.append(dy)
            diffs.append(math.hypot(dx, dy))
        
        print(f"    Samples matched: {len(diffs)}")
        print(f"    Mean offset:  dx={sum(diffs_x)/len(diffs_x):.3f}m  dy={sum(diffs_y)/len(diffs_y):.3f}m")
        print(f"    Mean distance: {sum(diffs)/len(diffs):.3f} m")
        print(f"    Max distance:  {max(diffs):.3f} m")
        print(f"    Min distance:  {min(diffs):.3f} m")
        
        # Check if offset is stable (good alignment) or drifting (bad alignment)
        first_half = diffs[:len(diffs)//2]
        second_half = diffs[len(diffs)//2:]
        mean_first = sum(first_half)/len(first_half) if first_half else 0
        mean_second = sum(second_half)/len(second_half) if second_half else 0
        
        if abs(mean_second - mean_first) > 0.5:
            print(f"    ⚠️  Offset is growing ({mean_first:.3f}m → {mean_second:.3f}m) — odom is drifting from GPS")
        else:
            print(f"    ✅  Offset is stable (first half: {mean_first:.3f}m, second half: {mean_second:.3f}m)")

    # === 5. Static segment check from GPS odom ===
    if gps_odom_t:
        print("\n" + "-" * 70)
        print("  [5] GPS Odom Static Noise (first 15s)")
        print("-" * 70)
        
        static_gx = [x for t, x in zip(gps_odom_t, gps_odom_x) if t - t0 < 15]
        static_gy = [y for t, y in zip(gps_odom_t, gps_odom_y) if t - t0 < 15]
        
        if len(static_gx) > 5:
            gx_mean = sum(static_gx) / len(static_gx)
            gy_mean = sum(static_gy) / len(static_gy)
            gx_std = (sum((x - gx_mean)**2 for x in static_gx) / len(static_gx)) ** 0.5
            gy_std = (sum((y - gy_mean)**2 for y in static_gy) / len(static_gy)) ** 0.5
            print(f"    Samples: {len(static_gx)}")
            print(f"    X std: {gx_std:.4f} m")
            print(f"    Y std: {gy_std:.4f} m")
            print(f"    2D std: {math.hypot(gx_std, gy_std):.4f} m")
            if math.hypot(gx_std, gy_std) < 0.05:
                print(f"    ✅  GPS odom stable during static (RTK fixed quality)")
            elif math.hypot(gx_std, gy_std) < 0.5:
                print(f"    ⚠️  Moderate GPS noise — float solution or poor fix")
            else:
                print(f"    🚨  Large GPS noise — check RTK status")
        else:
            print(f"    Not enough GPS odom samples in first 15s ({len(static_gx)})")

    # Export for plotting
    export = {
        "odom": {"t": [t-t0 for t in odom_t], "x": odom_x, "y": odom_y},
        "filtered": {"t": [t-t0 for t in filt_t], "x": filt_x, "y": filt_y},
        "gps_odom": {"t": [t-t0 for t in gps_odom_t], "x": gps_odom_x, "y": gps_odom_y},
    }
    if valid_rtk:
        lat0, lon0 = valid_rtk[0][1], valid_rtk[0][2]
        export["rtk_local"] = {
            "t": [t-t0 for t, _, _ in valid_rtk],
            "x": [latlon_to_xy(lat, lon, lat0, lon0)[0] for _, lat, lon in valid_rtk],
            "y": [latlon_to_xy(lat, lon, lat0, lon0)[1] for _, lat, lon in valid_rtk],
        }
    
    out_path = bag_path.replace('.bag', '_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(export, f)
    print(f"\n  Data exported to {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

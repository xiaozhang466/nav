# RTK Localization Workspace

ROS1 Noetic `catkin` workspace for Ranger wheel odometry, IMU, and UM980 RTK based positioning.

当前仓库已经收敛到一条可用的定位主链路：

- 轮式里程计 `vx + vy=0` + 修正后的 IMU `yaw + angular.z` -> local EKF -> `/odometry/filtered`
- RTK `fix` + local EKF -> `navsat_transform` -> `/odometry/gps`
- `/odometry/gps` 的全局 `x/y` + 修正后的 IMU yaw + local odom 的 `twist` -> `/odometry/global`
- `/odometry/filtered` 的局部 pose + GPS 平移 correction + 修正后的 IMU yaw -> `/odometry/global_nav`

`/odometry/global` 和 `/odometry/global_nav` 都由 [compose_global_odom.py](/home/sigu/RTK/44/src/um980_driver/scripts/compose_global_odom.py) 发布，其中 `rtk_map -> odom` 默认跟随 `/odometry/global_nav`。

## Current Status

- 当前方案已经可以稳定输出本地位姿和全局位姿，但暂不包含完整导航栈。
- 当前默认推荐的导航定位输出是 `/odometry/global_nav`。
- `/odometry/global` 更适合诊断：位置直接跟 RTK，航向直接跟修正后的 IMU。
- `navsat_transform` 默认使用修正后的 `/IMU_data_enu` 作为航向输入，默认 `yaw_bias_deg` 为 `11.7`。
- 底盘驱动需要单独启动，不包含在 `um980_driver` 的 launch 中。

## Layout

- [src/um980_driver/launch](/home/sigu/RTK/44/src/um980_driver/launch): 当前定位主入口
- [src/um980_driver/config](/home/sigu/RTK/44/src/um980_driver/config): EKF、RTK、`navsat_transform` 配置
- [src/um980_driver/scripts](/home/sigu/RTK/44/src/um980_driver/scripts): RTK ROS 节点、全局里程组合、分析工具
- [src/um980_driver/udev](/home/sigu/RTK/44/src/um980_driver/udev): 串口别名规则

## Build

```bash
cd /home/sigu/RTK/44
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Run

终端 1：

```bash
source /opt/ros/noetic/setup.bash
source /home/sigu/RTK/44/devel/setup.bash
roscore
```

终端 2，启动底盘：

```bash
source /opt/ros/noetic/setup.bash
source /home/sigu/RTK/44/devel/setup.bash
roslaunch ranger_bringup ranger_mini_v2.launch port_name:=can0 publish_odom_tf:=false
```

如果后面要切换车型，也可以继续用通用入口：

```bash
roslaunch ranger_bringup ranger.launch port_name:=can0 robot_model:=ranger_mini_v2 publish_odom_tf:=false
```

终端 3，启动定位链：

```bash
source /opt/ros/noetic/setup.bash
source /home/sigu/RTK/44/devel/setup.bash
roslaunch um980_driver global_localization.launch
```

这个入口会自动启动：

- [rtk_imu_bringup.launch](/home/sigu/RTK/44/src/um980_driver/launch/rtk_imu_bringup.launch)
- [imu_heading_correction.launch](/home/sigu/RTK/44/src/um980_driver/launch/imu_heading_correction.launch)
- [local_ekf.launch](/home/sigu/RTK/44/src/um980_driver/launch/local_ekf.launch)
- `navsat_transform`
- [compose_global_odom.py](/home/sigu/RTK/44/src/um980_driver/scripts/compose_global_odom.py) 的两路输出

## Recommended Outputs

- `/odometry/filtered`: 当前本地定位主输出
- `/odometry/global_nav`: 当前推荐的导航用全局定位输出
- `/odometry/global`: 纯 RTK + IMU 的诊断输出
- `rtk_map -> odom`: 当前默认跟随 `/odometry/global_nav`

## Main Topics

- `/ranger_base_node/odom`: 底盘轮式里程计
- `/IMU_data`: IMU 数据
- `/IMU_data_enu`: 经过“北为 0 -> 东为 0 + bias”修正后的 IMU 航向
- `/rtk/fix`: UM980 发布的 `NavSatFix`
- `/odometry/filtered`: local EKF 输出
- `/odometry/gps`: `navsat_transform` 输出
- `/odometry/global`: 纯 RTK+IMU 诊断输出
- `/odometry/global_nav`: 当前推荐的导航用全局定位输出

## Parameters

- [ekf_local.yaml](/home/sigu/RTK/44/src/um980_driver/config/ekf_local.yaml)
  当前使用轮式里程计 `vx + vy=0`，以及修正后的 IMU `yaw + angular.z`。
- [navsat_transform.yaml](/home/sigu/RTK/44/src/um980_driver/config/navsat_transform.yaml)
  当前默认使用修正后的 IMU 航向，`use_odometry_yaw: false`。
- [global_localization.launch](/home/sigu/RTK/44/src/um980_driver/launch/global_localization.launch)
  可以直接调 `imu_heading_yaw_bias_deg`、`gps_timeout`、`global_nav_correction_alpha`，也保留了切回 odom-yaw 的开关。

## Quick Checks

```bash
source /opt/ros/noetic/setup.bash
source /home/sigu/RTK/44/devel/setup.bash
rostopic echo -n 5 /rtk/fix_type
rostopic hz /odometry/filtered
rostopic hz /odometry/global_nav
```

正常情况下：

- `/rtk/fix_type` 应该优先看到 `rtk_fixed`
- `/odometry/filtered` 大约 `50 Hz`
- `/odometry/global_nav` 大约 `50 Hz`

## Rosbag Recording

```bash
source /opt/ros/noetic/setup.bash
source /home/sigu/RTK/44/devel/setup.bash
rosbag record -O /home/sigu/RTK/44/bags/global_nav_test_$(date +%Y%m%d_%H%M%S).bag \
  /IMU_data \
  /IMU_data_enu \
  /ranger_base_node/odom \
  /odometry/filtered \
  /odometry/gps \
  /odometry/global \
  /odometry/global_nav \
  /rtk/fix \
  /rtk/fix_type \
  /rtk/position_type \
  /rtk/rtkstatus_raw \
  /tf \
  /tf_static
```

## Useful Tools

- 分析 bag：

```bash
rosrun um980_driver analyze_nav_bag.py /path/to/file.bag --odom-topic /odometry/global
```

- 直接查看 UM980 输出：

```bash
rosrun um980_driver um980_monitor.py --config $(rospack find um980_driver)/config/rtk_client.ini
```

- 安装稳定串口别名：

```bash
sudo $(rospack find um980_driver)/scripts/install_rtk_udev.sh
```

## Notes

- 仓库里仍保留了 NTRIP 相关脚本，但当前这套定位方案不依赖它们。
- 如果要回退到旧方案，可以在 [global_localization.launch](/home/sigu/RTK/44/src/um980_driver/launch/global_localization.launch) 里设置 `navsat_use_odometry_yaw:=true global_yaw_source:=local`，并关闭 IMU 航向修正节点。
- 当前目标是稳定定位，后续如果接导航，优先使用 `/odometry/global_nav`。

cmake_minimum_required(VERSION 3.0.2)
project(um980_driver)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  costmap_2d
  geometry_msgs
  move_base_msgs
  nav_msgs
  nav_core
  pluginlib
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs costmap_2d geometry_msgs move_base_msgs nav_core nav_msgs pluginlib roscpp rospy sensor_msgs std_msgs std_srvs tf tf2_geometry_msgs tf2_ros visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(route_global_planner
  src/route_global_planner.cpp
)
target_link_libraries(route_global_planner
  ${catkin_LIBRARIES}
)

add_library(regulated_pure_pursuit_planner
  src/regulated_pure_pursuit_planner.cpp
)
target_link_libraries(regulated_pure_pursuit_planner
  ${catkin_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/alignment_estimator.py
  scripts/cmd_vel_smoother.py
  scripts/compose_global_odom.py
  scripts/global_localization_manager.py
  scripts/imu_heading_correction.py
  scripts/lidar_localization_adapter.py
  scripts/nav_health_monitor.py
  scripts/rtk_lidar_relocalizer.py
  scripts/static_transform_from_params.py
  scripts/tf_diagnostic_logger.py
  scripts/um980_ros_node.py
  scripts/waypoint_patrol_node.py
  scripts/waypoint_visualizer.py
  tools/analyze_nav_bag.py
  tools/estimate_fast_rtk_alignment.py
  tools/estimate_imu_covariance.py
  tools/estimate_odom_covariance.py
  tools/ntrip_rtk_bridge.py
  tools/rtk_diagnose.py
  tools/sample_waypoints_from_poses.py
  tools/um980_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  tools/install_rtk_udev.sh
  tools/start_rtk.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  maps
  poses
  waypoints
  udev
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS
  route_global_planner
  regulated_pure_pursuit_planner
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES
  route_global_planner_plugin.xml
  regulated_pure_pursuit_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

cmake_minimum_required(VERSION 3.0.2)
project(orchard_map_update)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  livox_ros_driver
  nav_msgs
  pcl_conversions
  pcl_ros
  rosbag
  roscpp
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2
  tf2_msgs
  tf2_ros
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(PCL REQUIRED COMPONENTS common filters io)
find_package(OpenMP QUIET)

catkin_package(
  INCLUDE_DIRS third_party/fast_livo2_frontend/include
  LIBRARIES fast_livo2_frontend
  CATKIN_DEPENDS
    geometry_msgs
    nav_msgs
    pcl_conversions
    pcl_ros
    rosbag
    roscpp
    sensor_msgs
    std_msgs
    std_srvs
    tf
    tf2
    tf2_msgs
    tf2_ros
    visualization_msgs
)

include_directories(
  third_party/fast_livo2_frontend/include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(fast_livo2_frontend
  third_party/fast_livo2_frontend/src/IMU_Processing.cpp
  third_party/fast_livo2_frontend/src/preprocess.cpp
  third_party/fast_livo2_frontend/src/voxel_map.cpp
)
target_compile_definitions(fast_livo2_frontend
  PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)
target_link_libraries(fast_livo2_frontend
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fast_livo2_frontend OpenMP::OpenMP_CXX)
endif()

add_executable(fast_livo2_lio_recorder_node
  src/fast_livo2_lio_recorder_node.cpp
)
add_dependencies(fast_livo2_lio_recorder_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(fast_livo2_lio_recorder_node
  fast_livo2_frontend
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fast_livo2_lio_recorder_node OpenMP::OpenMP_CXX)
endif()

add_executable(fast_livo2_bag_frame_exporter
  src/fast_livo2_bag_frame_exporter.cpp
)
add_dependencies(fast_livo2_bag_frame_exporter ${catkin_EXPORTED_TARGETS})
target_compile_definitions(fast_livo2_bag_frame_exporter
  PRIVATE ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)
target_link_libraries(fast_livo2_bag_frame_exporter
  fast_livo2_frontend
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(fast_livo2_bag_frame_exporter OpenMP::OpenMP_CXX)
endif()

install(TARGETS fast_livo2_frontend fast_livo2_lio_recorder_node fast_livo2_bag_frame_exporter
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  third_party
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

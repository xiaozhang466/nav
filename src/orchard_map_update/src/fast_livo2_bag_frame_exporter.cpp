#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <deque>
#include <limits>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/buffer_core.h>
#include <tf2/exceptions.h>
#include <tf2_msgs/TFMessage.h>

#include "IMU_Processing.h"
#include "preprocess.h"
#include "voxel_map.h"

#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

namespace fs = std::filesystem;

namespace
{
struct Options
{
  std::string bag_path;
  std::string output_dir;
  std::string pointcloud_topic = "/lslidar_point_cloud";
  std::string imu_topic = "/IMU_data";
  std::string map_frame = "map";
  std::string pose_frame = "lslidar";
  int lidar_type = VELO16;
  int scan_line = 16;
  int point_filter_num = 1;
  int min_points = 1;
  int max_frames = 0;
  int imu_init_frame_num = 30;
  int lio_max_layer = 2;
  int lio_max_iterations = 5;
  int lio_max_points_num = 50;
  int local_map_half_size = 100;
  double blind = 0.5;
  double lidar_time_offset = -0.02;
  double filter_size_surf = 0.1;
  double imu_init_buffer_time = 2.0;
  double imu_overlap_time = 0.02;
  double imu_end_tolerance = 0.02;
  double acc_cov = 0.5;
  double gyr_cov = 0.3;
  double b_acc_cov = 0.0001;
  double b_gyr_cov = 0.0001;
  double tf_cache_time = 86400.0;
  double lio_voxel_size = 0.5;
  double lio_min_eigen_value = 0.0025;
  double lio_sigma_num = 3.0;
  double lio_beam_err = 0.05;
  double lio_dept_err = 0.02;
  double local_map_sliding_thresh = 8.0;
  bool downsample = true;
  bool imu_en = true;
  bool gravity_align_en = true;
  bool require_pose = true;
  bool lio_state_update = true;
  bool local_map_sliding_en = false;
  std::vector<int> lio_layer_init_num = {5, 5, 5, 5, 5};
  V3D ext_t = V3D(0.0, 0.0, 0.3);
  M3D ext_r = M3D::Identity();
};

std::string boolText(bool value)
{
  return value ? "true" : "false";
}

bool parseBool(const std::string &value)
{
  return value == "1" || value == "true" || value == "True" || value == "yes" ||
         value == "on";
}

std::string cleanFrame(std::string frame)
{
  if (!frame.empty() && frame.front() == '/') {
    frame.erase(frame.begin());
  }
  return frame;
}

std::string bagStem(const std::string &bag_path)
{
  fs::path p(bag_path);
  std::string stem = p.stem().string();
  if (stem.empty()) {
    stem = "bag";
  }
  return stem;
}

fs::path defaultOutputDir(const std::string &bag_path)
{
  return fs::path(ROOT_DIR) / "update_datasets" / (bagStem(bag_path) + "_undistorted");
}

std::string formatSeq(uint64_t seq)
{
  std::ostringstream ss;
  ss << std::setw(6) << std::setfill('0') << seq;
  return ss.str();
}

std::string formatStamp(double stamp)
{
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(9) << stamp;
  return ss.str();
}

void printUsage()
{
  std::cerr
      << "Usage:\n"
      << "  rosrun orchard_map_update fast_livo2_bag_frame_exporter --bag BAG [options]\n\n"
      << "Options:\n"
      << "  --output DIR              Output dataset dir. Default: orchard_map_update/update_datasets/<bag>_undistorted\n"
      << "  --pointcloud_topic TOPIC  LiDAR PointCloud2 topic. Default: /lslidar_point_cloud\n"
      << "  --imu_topic TOPIC         IMU topic used for FAST-LIVO2 undistortion. Default: /IMU_data\n"
      << "  --map_frame FRAME         Pose target frame for TF lookup. Default: map\n"
      << "  --pose_frame FRAME        Pose source frame for TF lookup. Default: lslidar\n"
      << "  --require_pose true|false Skip frames without map->pose TF. Default: true\n"
      << "  --lidar_type N            FAST-LIVO2 lidar type. Default: 2 (VELO16)\n"
      << "  --scan_line N             LiDAR scan lines. Default: 16\n"
      << "  --point_filter_num N      FAST-LIVO2 point filter num. Default: 1\n"
      << "  --blind M                 Blind range in meters. Default: 0.5\n"
      << "  --lidar_time_offset S     Timestamp offset in seconds. Default: -0.02\n"
      << "  --imu_en true|false       Use IMU undistortion. Default: true\n"
      << "  --imu_init_frame_num N    FAST-LIVO2 IMU init frame count. Default: 30\n"
      << "  --imu_init_buffer_time S  IMU history before first LiDAR frame. Default: 2.0\n"
      << "  --imu_overlap_time S      IMU overlap before previous LIO update. Default: 0.02\n"
      << "  --imu_end_tolerance S     IMU tolerance after LiDAR frame end. Default: 0.02\n"
      << "  --tf_cache_time S         TF cache duration for offline lookup. Default: 86400\n"
      << "  --lio_state_update true|false Run VoxelMap LIO to maintain IMU state. Default: true\n"
      << "  --downsample true|false   Save VoxelGrid downsampled body cloud. Default: true\n"
      << "  --filter_size_surf M      VoxelGrid leaf size when downsample=true. Default: 0.1\n"
      << "  --min_points N            Skip output clouds smaller than N points. Default: 1\n"
      << "  --max_frames N            Stop after N saved frames. 0 means unlimited.\n";
}

void applyOption(Options &opts, const std::string &key, const std::string &value)
{
  if (key == "bag") {
    opts.bag_path = value;
  } else if (key == "output" || key == "output_dir") {
    opts.output_dir = (value == "auto") ? "" : value;
  } else if (key == "pointcloud_topic") {
    opts.pointcloud_topic = value;
  } else if (key == "imu_topic") {
    opts.imu_topic = value;
  } else if (key == "map_frame") {
    opts.map_frame = cleanFrame(value);
  } else if (key == "pose_frame") {
    opts.pose_frame = cleanFrame(value);
  } else if (key == "require_pose") {
    opts.require_pose = parseBool(value);
  } else if (key == "lidar_type") {
    opts.lidar_type = std::stoi(value);
  } else if (key == "scan_line") {
    opts.scan_line = std::stoi(value);
  } else if (key == "point_filter_num") {
    opts.point_filter_num = std::stoi(value);
  } else if (key == "blind") {
    opts.blind = std::stod(value);
  } else if (key == "lidar_time_offset") {
    opts.lidar_time_offset = std::stod(value);
  } else if (key == "imu_en") {
    opts.imu_en = parseBool(value);
  } else if (key == "imu_init_frame_num" || key == "imu_int_frame") {
    opts.imu_init_frame_num = std::stoi(value);
  } else if (key == "imu_init_buffer_time") {
    opts.imu_init_buffer_time = std::stod(value);
  } else if (key == "imu_overlap_time") {
    opts.imu_overlap_time = std::stod(value);
  } else if (key == "imu_end_tolerance") {
    opts.imu_end_tolerance = std::stod(value);
  } else if (key == "acc_cov") {
    opts.acc_cov = std::stod(value);
  } else if (key == "gyr_cov") {
    opts.gyr_cov = std::stod(value);
  } else if (key == "b_acc_cov") {
    opts.b_acc_cov = std::stod(value);
  } else if (key == "b_gyr_cov") {
    opts.b_gyr_cov = std::stod(value);
  } else if (key == "gravity_align_en") {
    opts.gravity_align_en = parseBool(value);
  } else if (key == "tf_cache_time") {
    opts.tf_cache_time = std::stod(value);
  } else if (key == "lio_state_update") {
    opts.lio_state_update = parseBool(value);
  } else if (key == "lio_voxel_size") {
    opts.lio_voxel_size = std::stod(value);
  } else if (key == "lio_min_eigen_value") {
    opts.lio_min_eigen_value = std::stod(value);
  } else if (key == "lio_sigma_num") {
    opts.lio_sigma_num = std::stod(value);
  } else if (key == "lio_beam_err") {
    opts.lio_beam_err = std::stod(value);
  } else if (key == "lio_dept_err") {
    opts.lio_dept_err = std::stod(value);
  } else if (key == "lio_max_layer") {
    opts.lio_max_layer = std::stoi(value);
  } else if (key == "lio_max_iterations") {
    opts.lio_max_iterations = std::stoi(value);
  } else if (key == "lio_max_points_num") {
    opts.lio_max_points_num = std::stoi(value);
  } else if (key == "downsample") {
    opts.downsample = parseBool(value);
  } else if (key == "filter_size_surf") {
    opts.filter_size_surf = std::stod(value);
  } else if (key == "min_points") {
    opts.min_points = std::stoi(value);
  } else if (key == "max_frames") {
    opts.max_frames = std::stoi(value);
  }
}

Options parseOptions(int argc, char **argv)
{
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    }

    std::string key;
    std::string value;
    if (arg.rfind("--", 0) == 0) {
      std::string body = arg.substr(2);
      const size_t eq = body.find('=');
      if (eq != std::string::npos) {
        key = body.substr(0, eq);
        value = body.substr(eq + 1);
      } else {
        key = body;
        if (i + 1 >= argc) {
          throw std::runtime_error("missing value for --" + key);
        }
        value = argv[++i];
      }
    } else if (!arg.empty() && arg[0] == '_') {
      const size_t sep = arg.find(":=");
      if (sep == std::string::npos) {
        continue;
      }
      key = arg.substr(1, sep - 1);
      value = arg.substr(sep + 2);
    } else {
      continue;
    }
    applyOption(opts, key, value);
  }
  return opts;
}

void writeMetadata(const fs::path &output_dir,
                   const Options &opts,
                   uint64_t total_clouds,
                   uint64_t total_imus,
                   uint64_t dynamic_tf_count,
                   uint64_t static_tf_count,
                   uint64_t saved_frames,
                   uint64_t skipped_frames,
                   uint64_t imu_wait_frames,
                   uint64_t pose_missing_frames)
{
  std::ofstream out((output_dir / "metadata.yaml").string(), std::ios::out);
  out << "format: fast_livo2_bag_lio_corrected_frames\n";
  out << "bag: " << opts.bag_path << "\n";
  out << "pointcloud_topic: " << opts.pointcloud_topic << "\n";
  out << "imu_topic: " << opts.imu_topic << "\n";
  out << "map_frame: " << opts.map_frame << "\n";
  out << "pose_frame: " << opts.pose_frame << "\n";
  out << "pose_source: bag_tf\n";
  out << "pose_file: lidar_poses.txt\n";
  out << "output_dir: " << output_dir.string() << "\n";
  out << "output_cloud: fast_livo2_preprocess_imu_undistorted";
  if (opts.lio_state_update) {
    out << "_lio_state_corrected";
  }
  if (opts.downsample) {
    out << "_downsampled";
  }
  out << "\n";
  out << "total_clouds: " << total_clouds << "\n";
  out << "total_imus: " << total_imus << "\n";
  out << "dynamic_tf_count: " << dynamic_tf_count << "\n";
  out << "static_tf_count: " << static_tf_count << "\n";
  out << "saved_frames: " << saved_frames << "\n";
  out << "skipped_frames: " << skipped_frames << "\n";
  out << "imu_wait_frames: " << imu_wait_frames << "\n";
  out << "pose_missing_frames: " << pose_missing_frames << "\n";
  out << "preprocess:\n";
  out << "  lidar_type: " << opts.lidar_type << "\n";
  out << "  scan_line: " << opts.scan_line << "\n";
  out << "  point_filter_num: " << opts.point_filter_num << "\n";
  out << "  blind: " << opts.blind << "\n";
  out << "  lidar_time_offset: " << opts.lidar_time_offset << "\n";
  out << "  downsample: " << boolText(opts.downsample) << "\n";
  out << "  filter_size_surf: " << opts.filter_size_surf << "\n";
  out << "imu:\n";
  out << "  imu_en: " << boolText(opts.imu_en) << "\n";
  out << "  imu_init_frame_num: " << opts.imu_init_frame_num << "\n";
  out << "  imu_init_buffer_time: " << opts.imu_init_buffer_time << "\n";
  out << "  imu_overlap_time: " << opts.imu_overlap_time << "\n";
  out << "  imu_end_tolerance: " << opts.imu_end_tolerance << "\n";
  out << "  acc_cov: " << opts.acc_cov << "\n";
  out << "  gyr_cov: " << opts.gyr_cov << "\n";
  out << "  b_acc_cov: " << opts.b_acc_cov << "\n";
  out << "  b_gyr_cov: " << opts.b_gyr_cov << "\n";
  out << "  gravity_align_en: " << boolText(opts.gravity_align_en) << "\n";
  out << "tf:\n";
  out << "  cache_time: " << opts.tf_cache_time << "\n";
  out << "  require_pose: " << boolText(opts.require_pose) << "\n";
  out << "lio:\n";
  out << "  state_update: " << boolText(opts.lio_state_update) << "\n";
  out << "  voxel_size: " << opts.lio_voxel_size << "\n";
  out << "  min_eigen_value: " << opts.lio_min_eigen_value << "\n";
  out << "  sigma_num: " << opts.lio_sigma_num << "\n";
  out << "  beam_err: " << opts.lio_beam_err << "\n";
  out << "  dept_err: " << opts.lio_dept_err << "\n";
  out << "  max_layer: " << opts.lio_max_layer << "\n";
  out << "  max_iterations: " << opts.lio_max_iterations << "\n";
  out << "  max_points_num: " << opts.lio_max_points_num << "\n";
}

VoxelMapConfig makeVoxelMapConfig(const Options &opts)
{
  VoxelMapConfig config;
  config.is_pub_plane_map_ = false;
  config.max_layer_ = opts.lio_max_layer;
  config.max_voxel_size_ = opts.lio_voxel_size;
  config.planner_threshold_ = opts.lio_min_eigen_value;
  config.sigma_num_ = opts.lio_sigma_num;
  config.beam_err_ = opts.lio_beam_err;
  config.dept_err_ = opts.lio_dept_err;
  config.layer_init_num_ = opts.lio_layer_init_num;
  config.max_points_num_ = opts.lio_max_points_num;
  config.max_iterations_ = opts.lio_max_iterations;
  config.map_sliding_en = opts.local_map_sliding_en;
  config.half_map_size = opts.local_map_half_size;
  config.sliding_thresh = opts.local_map_sliding_thresh;
  return config;
}

void releaseVoxelMap(VoxelMapManagerPtr &voxelmap_manager)
{
  if (!voxelmap_manager) {
    return;
  }
  for (auto &entry : voxelmap_manager->voxel_map_) {
    delete entry.second;
  }
  voxelmap_manager->voxel_map_.clear();
}

std::vector<sensor_msgs::ImuConstPtr> loadImuMessages(rosbag::Bag &bag,
                                                       const std::string &imu_topic)
{
  std::vector<sensor_msgs::ImuConstPtr> imus;
  rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{imu_topic}));
  imus.reserve(view.size());
  for (const rosbag::MessageInstance &instance : view) {
    sensor_msgs::ImuConstPtr msg = instance.instantiate<sensor_msgs::Imu>();
    if (msg) {
      imus.push_back(msg);
    }
  }
  std::sort(imus.begin(), imus.end(), [](const auto &a, const auto &b) {
    return a->header.stamp < b->header.stamp;
  });
  return imus;
}

std::deque<sensor_msgs::ImuConstPtr> collectImu(
    const std::vector<sensor_msgs::ImuConstPtr> &imus,
    double start,
    double end)
{
  std::deque<sensor_msgs::ImuConstPtr> result;
  auto it = std::lower_bound(imus.begin(), imus.end(), start,
                             [](const sensor_msgs::ImuConstPtr &imu, double t) {
                               return imu->header.stamp.toSec() < t;
                             });
  for (; it != imus.end(); ++it) {
    const double t = (*it)->header.stamp.toSec();
    if (t > end) {
      break;
    }
    result.push_back(*it);
  }
  return result;
}

void loadTfMessages(rosbag::Bag &bag,
                    tf2::BufferCore &tf_buffer,
                    uint64_t &dynamic_tf_count,
                    uint64_t &static_tf_count)
{
  dynamic_tf_count = 0;
  static_tf_count = 0;
  rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{"/tf", "/tf_static"}));
  for (const rosbag::MessageInstance &instance : view) {
    tf2_msgs::TFMessageConstPtr msg = instance.instantiate<tf2_msgs::TFMessage>();
    if (!msg) {
      continue;
    }
    const bool is_static = instance.getTopic() == "/tf_static" ||
                           instance.getTopic() == "tf_static";
    for (const auto &transform : msg->transforms) {
      try {
        tf_buffer.setTransform(transform, "bag", is_static);
        if (is_static) {
          ++static_tf_count;
        } else {
          ++dynamic_tf_count;
        }
      } catch (const tf2::TransformException &exc) {
        ROS_WARN_STREAM_THROTTLE(5.0, "failed to load TF from bag: " << exc.what());
      }
    }
  }
}

bool lookupPose(tf2::BufferCore &tf_buffer,
                const Options &opts,
                double stamp_sec,
                geometry_msgs::TransformStamped &pose,
                std::string &error)
{
  try {
    pose = tf_buffer.lookupTransform(
        opts.map_frame, opts.pose_frame, ros::Time(stamp_sec));
    return true;
  } catch (const tf2::TransformException &exc) {
    error = exc.what();
    return false;
  }
}

void gravityAlignment(ImuProcess &imu_process,
                      StatesGroup &state,
                      bool &gravity_align_finished)
{
  if (!imu_process.imu_need_init && !gravity_align_finished) {
    V3D ez(0, 0, -1), gz(state.gravity);
    Eigen::Quaterniond g_q_i0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
    M3D g_r_i0 = g_q_i0.toRotationMatrix();
    state.pos_end = g_r_i0 * state.pos_end;
    state.rot_end = g_r_i0 * state.rot_end;
    state.vel_end = g_r_i0 * state.vel_end;
    state.gravity = g_r_i0 * state.gravity;
    gravity_align_finished = true;
    ROS_INFO("FAST-LIVO2 bag exporter gravity alignment finished");
  }
}
}  // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fast_livo2_bag_frame_exporter",
            ros::init_options::AnonymousName | ros::init_options::NoSigintHandler);

  Options opts;
  try {
    opts = parseOptions(argc, argv);
  } catch (const std::exception &exc) {
    std::cerr << "Argument error: " << exc.what() << "\n";
    printUsage();
    return 2;
  }

  if (opts.bag_path.empty()) {
    printUsage();
    return 2;
  }

  fs::path output_dir = opts.output_dir.empty() ? defaultOutputDir(opts.bag_path)
                                                : fs::path(opts.output_dir);
  fs::path pcd_dir = output_dir / "pcd";
  fs::create_directories(pcd_dir);

  Preprocess preprocess;
  preprocess.N_SCANS = opts.scan_line;
  preprocess.set(false, opts.lidar_type, opts.blind, opts.point_filter_num);

  ImuProcess imu_process;
  imu_process.lidar_type = opts.lidar_type;
  imu_process.set_extrinsic(opts.ext_t, opts.ext_r);
  imu_process.set_gyr_cov_scale(V3D(opts.gyr_cov, opts.gyr_cov, opts.gyr_cov));
  imu_process.set_acc_cov_scale(V3D(opts.acc_cov, opts.acc_cov, opts.acc_cov));
  imu_process.set_gyr_bias_cov(V3D(opts.b_gyr_cov, opts.b_gyr_cov, opts.b_gyr_cov));
  imu_process.set_acc_bias_cov(V3D(opts.b_acc_cov, opts.b_acc_cov, opts.b_acc_cov));
  imu_process.set_imu_init_frame_num(opts.imu_init_frame_num);
  imu_process.disable_exposure_est();
  if (!opts.imu_en) {
    imu_process.disable_imu();
  }
  StatesGroup state;
  StatesGroup state_propagat;
  bool first_lidar_seen = false;
  bool gravity_align_finished = false;
  bool lidar_map_inited = false;
  double first_lidar_time = 0.0;
  double last_lio_update_time = -1.0;

  pcl::VoxelGrid<PointType> downsample_filter;
  downsample_filter.setLeafSize(
      opts.filter_size_surf, opts.filter_size_surf, opts.filter_size_surf);
  VoxelMapConfig voxel_config = makeVoxelMapConfig(opts);
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_storage;
  VoxelMapManagerPtr voxelmap_manager(new VoxelMapManager(voxel_config, voxel_map_storage));
  voxelmap_manager->extR_ = opts.ext_r;
  voxelmap_manager->extT_ = opts.ext_t;
  pcl::PointCloud<pcl::PointXYZI>::Ptr feats_down_world(new pcl::PointCloud<pcl::PointXYZI>());

  std::ofstream frames_csv((output_dir / "frames.csv").string(), std::ios::out);
  frames_csv << "seq,stamp,frame_begin,frame_end,pcd,frame_id,raw_points,"
                "preprocessed_points,undistorted_points,output_points,imu_count,"
                "lio_effective_features,lidar_map_inited,pose_status,x,y,z,qx,qy,qz,qw,"
                "first_offset_ms,last_offset_ms\n";
  std::ofstream pose_file((output_dir / "lidar_poses.txt").string(), std::ios::out);

  uint64_t total_clouds = 0;
  uint64_t total_imus = 0;
  uint64_t dynamic_tf_count = 0;
  uint64_t static_tf_count = 0;
  uint64_t saved_frames = 0;
  uint64_t skipped_frames = 0;
  uint64_t imu_wait_frames = 0;
  uint64_t pose_missing_frames = 0;

  try {
    rosbag::Bag bag;
    bag.open(opts.bag_path, rosbag::bagmode::Read);
    std::vector<sensor_msgs::ImuConstPtr> imu_messages;
    if (opts.imu_en) {
      imu_messages = loadImuMessages(bag, opts.imu_topic);
      total_imus = imu_messages.size();
      if (imu_messages.empty()) {
        throw std::runtime_error("no IMU messages found on topic " + opts.imu_topic);
      }
      ROS_INFO_STREAM("loaded " << total_imus << " IMU messages from " << opts.imu_topic);
    }

    tf2::BufferCore tf_buffer(ros::Duration(opts.tf_cache_time));
    loadTfMessages(bag, tf_buffer, dynamic_tf_count, static_tf_count);
    ROS_INFO_STREAM("loaded TF messages: dynamic=" << dynamic_tf_count
                    << " static=" << static_tf_count);

    rosbag::View view(bag, rosbag::TopicQuery(std::vector<std::string>{opts.pointcloud_topic}));

    pcl::PCDWriter writer;
    for (const rosbag::MessageInstance &instance : view) {
      sensor_msgs::PointCloud2ConstPtr msg = instance.instantiate<sensor_msgs::PointCloud2>();
      if (!msg) {
        continue;
      }
      ++total_clouds;

      PointCloudXYZI::Ptr preprocessed(new PointCloudXYZI());
      try {
        preprocess.process(msg, preprocessed);
      } catch (const std::exception &exc) {
        ++skipped_frames;
        ROS_WARN_STREAM("preprocess failed at message " << total_clouds << ": " << exc.what());
        continue;
      }

      if (!preprocessed || preprocessed->empty()) {
        ++skipped_frames;
        continue;
      }

      const ros::Time stamp = msg->header.stamp.isZero() ? instance.getTime()
                                                         : msg->header.stamp;
      const double frame_begin = stamp.toSec() + opts.lidar_time_offset;
      const double first_offset_ms = preprocessed->points.front().curvature;
      const double last_offset_ms = preprocessed->points.back().curvature;
      const double frame_end = frame_begin + last_offset_ms / 1000.0;

      PointCloudXYZI::Ptr undistorted(new PointCloudXYZI());
      std::deque<sensor_msgs::ImuConstPtr> imu_window;
      if (opts.imu_en) {
        const double imu_start = last_lio_update_time > 0.0
                                     ? std::max(0.0, last_lio_update_time - opts.imu_overlap_time)
                                     : std::max(0.0, frame_begin - opts.imu_init_buffer_time);
        const double imu_end = frame_end + opts.imu_end_tolerance;
        imu_window = collectImu(imu_messages, imu_start, imu_end);
        if (imu_window.empty()) {
          ++skipped_frames;
          ++imu_wait_frames;
          continue;
        }

        LidarMeasureGroup lidar_meas;
        lidar_meas.lidar_frame_beg_time = frame_begin;
        lidar_meas.lidar_frame_end_time = frame_end;
        lidar_meas.last_lio_update_time = last_lio_update_time;
        lidar_meas.lidar = preprocessed;
        lidar_meas.pcl_proc_cur = preprocessed;
        lidar_meas.lio_vio_flg = LIO;
        lidar_meas.lidar_scan_index_now = 0;

        MeasureGroup meas;
        meas.lio_time = frame_end;
        meas.imu = imu_window;
        lidar_meas.measures.clear();
        lidar_meas.measures.push_back(meas);

        if (!first_lidar_seen) {
          first_lidar_seen = true;
          first_lidar_time = frame_end;
          imu_process.first_lidar_time = first_lidar_time;
          ROS_INFO("FAST-LIVO2 bag exporter first lidar frame: %.6f", first_lidar_time);
        }

        imu_process.Process2(lidar_meas, state, undistorted);
        if (opts.gravity_align_en) {
          gravityAlignment(imu_process, state, gravity_align_finished);
        }
        last_lio_update_time = lidar_meas.last_lio_update_time;

        if (!undistorted || undistorted->empty()) {
          ++skipped_frames;
          ++imu_wait_frames;
          continue;
        }
      } else {
        undistorted = preprocessed;
      }

      PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
      downsample_filter.setInputCloud(undistorted);
      downsample_filter.filter(*feats_down_body);
      if (!feats_down_body || feats_down_body->empty()) {
        ++skipped_frames;
        continue;
      }

      if (opts.lio_state_update) {
        state_propagat = state;
        voxelmap_manager->state_ = state;
        voxelmap_manager->feats_undistort_ = undistorted;
        voxelmap_manager->feats_down_body_ = feats_down_body;
        voxelmap_manager->TransformLidar(state.rot_end, state.pos_end, feats_down_body, feats_down_world);
        voxelmap_manager->feats_down_world_ = feats_down_world;
        voxelmap_manager->feats_down_size_ = static_cast<int>(feats_down_body->size());

        if (!lidar_map_inited) {
          lidar_map_inited = true;
          voxelmap_manager->BuildVoxelMap();
        }

        voxelmap_manager->StateEstimation(state_propagat);
        state = voxelmap_manager->state_;

        pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>());
        voxelmap_manager->TransformLidar(state.rot_end, state.pos_end, feats_down_body, world_lidar);
        const size_t n = std::min(world_lidar->points.size(), voxelmap_manager->pv_list_.size());
        for (size_t i = 0; i < n; ++i) {
          voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x,
              world_lidar->points[i].y,
              world_lidar->points[i].z;
          if (i < voxelmap_manager->cross_mat_list_.size() &&
              i < voxelmap_manager->body_cov_list_.size()) {
            M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
            M3D var = voxelmap_manager->body_cov_list_[i];
            var = (state.rot_end * opts.ext_r) * var * (state.rot_end * opts.ext_r).transpose() +
                  (-point_crossmat) * state.cov.block<3, 3>(0, 0) *
                      (-point_crossmat).transpose() +
                  state.cov.block<3, 3>(3, 3);
            voxelmap_manager->pv_list_[i].var = var;
          }
        }
        voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
        if (voxelmap_manager->config_setting_.map_sliding_en) {
          voxelmap_manager->mapSliding();
        }
      }

      geometry_msgs::TransformStamped pose_tf;
      std::string pose_error;
      const bool pose_ok = lookupPose(tf_buffer, opts, frame_end, pose_tf, pose_error);
      if (!pose_ok) {
        ++pose_missing_frames;
        if (opts.require_pose) {
          ++skipped_frames;
          ROS_WARN_STREAM_THROTTLE(5.0, "pose unavailable at " << frame_end
                                   << " for " << opts.map_frame << "->"
                                   << opts.pose_frame << ": " << pose_error);
          continue;
        }
      }

      PointCloudXYZI::Ptr output_cloud = opts.downsample ? feats_down_body : undistorted;

      if (!output_cloud || static_cast<int>(output_cloud->size()) < opts.min_points) {
        ++skipped_frames;
        continue;
      }

      const std::string file_name = formatSeq(saved_frames) + ".pcd";
      const fs::path pcd_path = pcd_dir / file_name;
      if (writer.writeBinary(pcd_path.string(), *output_cloud) != 0) {
        ++skipped_frames;
        ROS_WARN_STREAM("failed to write " << pcd_path);
        continue;
      }

      const uint64_t raw_points = static_cast<uint64_t>(msg->width) *
                                  static_cast<uint64_t>(msg->height);
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const double x = pose_ok ? pose_tf.transform.translation.x : nan;
      const double y = pose_ok ? pose_tf.transform.translation.y : nan;
      const double z = pose_ok ? pose_tf.transform.translation.z : nan;
      const double qx = pose_ok ? pose_tf.transform.rotation.x : nan;
      const double qy = pose_ok ? pose_tf.transform.rotation.y : nan;
      const double qz = pose_ok ? pose_tf.transform.rotation.z : nan;
      const double qw = pose_ok ? pose_tf.transform.rotation.w : nan;
      frames_csv << saved_frames << ","
                 << formatStamp(stamp.toSec()) << ","
                 << formatStamp(frame_begin) << ","
                 << formatStamp(frame_end) << ","
                 << file_name << ","
                 << msg->header.frame_id << ","
                 << raw_points << ","
                 << preprocessed->size() << ","
                 << undistorted->size() << ","
                 << output_cloud->size() << ","
                 << imu_window.size() << ","
                 << (opts.lio_state_update ? voxelmap_manager->effct_feat_num_ : 0) << ","
                 << (lidar_map_inited ? "true" : "false") << ","
                 << (pose_ok ? "ok" : "missing") << ","
                 << x << ","
                 << y << ","
                 << z << ","
                 << qx << ","
                 << qy << ","
                 << qz << ","
                 << qw << ","
                 << first_offset_ms << ","
                 << last_offset_ms << "\n";

      if (pose_ok) {
        pose_file << file_name << " "
                  << formatStamp(frame_end) << " "
                  << std::fixed << std::setprecision(9)
                  << x << " " << y << " " << z << " "
                  << qx << " " << qy << " " << qz << " " << qw << "\n";
      }

      ++saved_frames;
      if (saved_frames % 100 == 0) {
        ROS_INFO_STREAM("exported " << saved_frames << " IMU-undistorted frames");
      }
      if (opts.max_frames > 0 && static_cast<int>(saved_frames) >= opts.max_frames) {
        break;
      }
    }
    bag.close();
  } catch (const std::exception &exc) {
    std::cerr << "Failed to export bag frames: " << exc.what() << "\n";
    writeMetadata(output_dir, opts, total_clouds, total_imus, dynamic_tf_count,
                  static_tf_count, saved_frames, skipped_frames, imu_wait_frames,
                  pose_missing_frames);
    releaseVoxelMap(voxelmap_manager);
    return 1;
  }

  frames_csv.close();
  pose_file.close();
  writeMetadata(output_dir, opts, total_clouds, total_imus, dynamic_tf_count,
                static_tf_count, saved_frames, skipped_frames, imu_wait_frames,
                pose_missing_frames);
  releaseVoxelMap(voxelmap_manager);

  ROS_INFO_STREAM("FAST-LIVO2 bag frame export done: output=" << output_dir
                  << " clouds=" << total_clouds
                  << " imus=" << total_imus
                  << " saved=" << saved_frames
                  << " skipped=" << skipped_frames
                  << " pose_missing=" << pose_missing_frames);
  return 0;
}

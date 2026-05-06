#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/TransformStamped.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include "IMU_Processing.h"
#include "preprocess.h"
#include "voxel_map.h"

namespace fs = std::filesystem;

namespace
{
std::string cleanFrame(std::string frame)
{
  if (!frame.empty() && frame.front() == '/') {
    frame.erase(frame.begin());
  }
  return frame;
}

std::string expandUser(const std::string &path)
{
  if (path.empty() || path[0] != '~') {
    return path;
  }
  const char *home = std::getenv("HOME");
  if (home == nullptr) {
    return path;
  }
  if (path.size() == 1) {
    return std::string(home);
  }
  if (path[1] == '/') {
    return std::string(home) + path.substr(1);
  }
  return path;
}

}  // namespace

class FastLivo2LioRecorder
{
public:
  FastLivo2LioRecorder()
      : pnh_("~"), tf_listener_(tf_buffer_)
  {
    loadParams();
    resetFrontend();

    status_pub_ = pnh_.advertise<std_msgs::String>("status", 5, true);
    cloud_sub_ = nh_.subscribe(pointcloud_topic_, 5, &FastLivo2LioRecorder::cloudCallback, this);
    imu_sub_ = nh_.subscribe(imu_topic_, 400, &FastLivo2LioRecorder::imuCallback, this);

    start_srv_ = pnh_.advertiseService("start", &FastLivo2LioRecorder::handleStart, this);
    stop_srv_ = pnh_.advertiseService("stop", &FastLivo2LioRecorder::handleStop, this);
    status_srv_ = pnh_.advertiseService("status", &FastLivo2LioRecorder::handleStatus, this);
    status_timer_ = nh_.createTimer(ros::Duration(1.0), &FastLivo2LioRecorder::statusTimer, this);

    if (auto_start_) {
      std::string message;
      if (!startRecording(message)) {
        ROS_ERROR_STREAM("fast_livo2_lio_recorder auto_start failed: " << message);
      }
    }

    ROS_INFO_STREAM("FAST-LIVO2 LIO recorder ready: cloud=" << pointcloud_topic_
                    << " imu=" << imu_topic_ << " pose=" << map_frame_ << "->"
                    << pose_frame_);
  }

  ~FastLivo2LioRecorder()
  {
    closeSessionFiles();
    releaseVoxelMap();
  }

private:
  struct ExternalPose
  {
    double stamp = 0.0;
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  };

  void loadParams()
  {
    pnh_.param<std::string>("pointcloud_topic", pointcloud_topic_, "/lslidar_point_cloud");
    pnh_.param<std::string>("imu_topic", imu_topic_, "/IMU_data");
    pnh_.param<std::string>("session_root", session_root_, "~/.ros/map_update_sessions");
    pnh_.param<std::string>("session_prefix", session_prefix_, "fast_livo2_lio");
    pnh_.param<bool>("auto_start", auto_start_, false);

    pnh_.param<std::string>("map_frame", map_frame_, "map");
    pnh_.param<std::string>("pose_frame", pose_frame_, "laser_link");
    pnh_.param<std::string>("pose_fallback_frame", pose_fallback_frame_, "base_link");
    map_frame_ = cleanFrame(map_frame_);
    pose_frame_ = cleanFrame(pose_frame_);
    pose_fallback_frame_ = cleanFrame(pose_fallback_frame_);
    pnh_.param<double>("tf_timeout", tf_timeout_, 0.05);
    pnh_.param<bool>("allow_latest_tf_fallback", allow_latest_tf_fallback_, false);
    pnh_.param<bool>("pose_fallback_from_extrinsic", pose_fallback_from_extrinsic_, true);

    pnh_.param<int>("preprocess/lidar_type", lidar_type_, VELO16);
    pnh_.param<int>("preprocess/scan_line", scan_line_, 16);
    pnh_.param<int>("preprocess/point_filter_num", point_filter_num_, 1);
    pnh_.param<double>("preprocess/blind", blind_, 0.5);
    pnh_.param<double>("preprocess/filter_size_surf", filter_size_surf_, 0.1);
    pnh_.param<double>("time_offset/lidar_time_offset", lidar_time_offset_, 0.0);

    pnh_.param<bool>("imu/imu_en", imu_en_, true);
    pnh_.param<int>("imu/imu_int_frame", imu_init_frame_num_, 20);
    pnh_.param<double>("imu/acc_cov", acc_cov_, 0.5);
    pnh_.param<double>("imu/gyr_cov", gyr_cov_, 0.3);
    pnh_.param<double>("imu/b_acc_cov", b_acc_cov_, 0.0001);
    pnh_.param<double>("imu/b_gyr_cov", b_gyr_cov_, 0.0001);
    pnh_.param<bool>("uav/gravity_align_en", gravity_align_en_, true);

    pnh_.param<double>("sync/imu_init_buffer_time", imu_init_buffer_time_, 2.0);
    pnh_.param<double>("sync/imu_overlap_time", imu_overlap_time_, 0.02);
    pnh_.param<double>("sync/imu_prune_keep_time", imu_prune_keep_time_, 1.0);
    pnh_.param<double>("sync/imu_end_tolerance", imu_end_tolerance_, 0.02);
    pnh_.param<int>("sync/max_imu_buffer_size", max_imu_buffer_size_, 5000);

    pnh_.param<int>("frames/min_points", min_frame_points_, 300);

    std::vector<double> extrin_t = {0.0, 0.0, 0.3};
    std::vector<double> extrin_r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    pnh_.param<std::vector<double>>("extrin_calib/extrinsic_T", extrin_t, extrin_t);
    pnh_.param<std::vector<double>>("extrin_calib/extrinsic_R", extrin_r, extrin_r);
    if (extrin_t.size() != 3 || extrin_r.size() != 9) {
      ROS_WARN("Invalid extrin_calib values, using identity rotation and [0,0,0.3]");
      extrin_t = {0.0, 0.0, 0.3};
      extrin_r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    }
    ext_t_ << extrin_t[0], extrin_t[1], extrin_t[2];
    ext_r_ << extrin_r[0], extrin_r[1], extrin_r[2],
              extrin_r[3], extrin_r[4], extrin_r[5],
              extrin_r[6], extrin_r[7], extrin_r[8];

    loadVoxelConfig(pnh_, voxel_config_);
  }

  void resetFrontend()
  {
    releaseVoxelMap();

    preprocess_.reset(new Preprocess());
    preprocess_->N_SCANS = scan_line_;
    preprocess_->set(false, lidar_type_, blind_, point_filter_num_);

    imu_process_.reset(new ImuProcess());
    imu_process_->lidar_type = lidar_type_;
    imu_process_->set_extrinsic(ext_t_, ext_r_);
    imu_process_->set_gyr_cov_scale(V3D(gyr_cov_, gyr_cov_, gyr_cov_));
    imu_process_->set_acc_cov_scale(V3D(acc_cov_, acc_cov_, acc_cov_));
    imu_process_->set_gyr_bias_cov(V3D(b_gyr_cov_, b_gyr_cov_, b_gyr_cov_));
    imu_process_->set_acc_bias_cov(V3D(b_acc_cov_, b_acc_cov_, b_acc_cov_));
    imu_process_->set_imu_init_frame_num(imu_init_frame_num_);
    imu_process_->disable_exposure_est();
    if (!imu_en_) {
      imu_process_->disable_imu();
    }

    voxel_map_storage_.clear();
    voxelmap_manager_.reset(new VoxelMapManager(voxel_config_, voxel_map_storage_));
    voxelmap_manager_->extR_ = ext_r_;
    voxelmap_manager_->extT_ = ext_t_;
    voxelmap_manager_->voxel_map_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("map_update_planes", 1);

    downsample_filter_.setLeafSize(filter_size_surf_, filter_size_surf_, filter_size_surf_);
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    state_ = StatesGroup();
    state_propagat_ = StatesGroup();
    lidar_map_inited_ = false;
    gravity_align_finished_ = false;
    first_lidar_seen_ = false;
    last_lio_update_time_ = -1.0;
  }

  void releaseVoxelMap()
  {
    if (voxelmap_manager_) {
      for (auto &entry : voxelmap_manager_->voxel_map_) {
        delete entry.second;
      }
      voxelmap_manager_->voxel_map_.clear();
    }
  }

  void imuCallback(const sensor_msgs::ImuConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    imu_buffer_.push_back(msg);
    while (static_cast<int>(imu_buffer_.size()) > max_imu_buffer_size_) {
      imu_buffer_.pop_front();
    }
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!active_) {
        return;
      }
      ++cloud_count_;
    }

    PointCloudXYZI::Ptr raw_cloud(new PointCloudXYZI());
    try {
      preprocess_->process(msg, raw_cloud);
    } catch (const std::exception &exc) {
      rejectFrame(std::string("preprocess_failed:") + exc.what());
      return;
    }

    if (!raw_cloud || raw_cloud->empty()) {
      skipFrame("empty_preprocessed_cloud");
      return;
    }

    const double frame_beg = msg->header.stamp.toSec() + lidar_time_offset_;
    const double frame_end = frame_beg + raw_cloud->points.back().curvature / 1000.0;
    auto imu_msgs = collectImu(frame_beg, frame_end);
    if (imu_en_ && imu_msgs.empty()) {
      skipFrame("waiting_for_imu");
      return;
    }

    if (!processLio(raw_cloud, frame_beg, frame_end, imu_msgs)) {
      skipFrame("lio_initializing_or_no_output");
      return;
    }

    ExternalPose pose;
    std::string pose_error;
    if (!lookupExternalPose(frame_end, pose, pose_error)) {
      skipFrame("pose_unavailable:" + pose_error);
      return;
    }

    if (static_cast<int>(feats_down_body_->size()) < min_frame_points_) {
      rejectFrame("too_few_downsampled_points:" + std::to_string(feats_down_body_->size()));
      return;
    }

    if (!saveFrame(frame_end, pose)) {
      return;
    }
  }

  std::deque<sensor_msgs::ImuConstPtr> collectImu(double frame_beg, double frame_end)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const double start = last_lio_update_time_ > 0.0
                             ? std::max(0.0, last_lio_update_time_ - imu_overlap_time_)
                             : std::max(0.0, frame_beg - imu_init_buffer_time_);
    const double end = frame_end + imu_end_tolerance_;

    std::deque<sensor_msgs::ImuConstPtr> result;
    for (const auto &imu : imu_buffer_) {
      const double t = imu->header.stamp.toSec();
      if (t >= start && t <= end) {
        result.push_back(imu);
      }
    }

    const double prune_before = frame_end - imu_prune_keep_time_;
    while (imu_buffer_.size() > 2 && imu_buffer_.front()->header.stamp.toSec() < prune_before) {
      imu_buffer_.pop_front();
    }
    return result;
  }

  bool processLio(const PointCloudXYZI::Ptr &raw_cloud,
                  double frame_beg,
                  double frame_end,
                  const std::deque<sensor_msgs::ImuConstPtr> &imu_msgs)
  {
    LidarMeasureGroup lidar_meas;
    lidar_meas.lidar_frame_beg_time = frame_beg;
    lidar_meas.lidar_frame_end_time = frame_end;
    lidar_meas.last_lio_update_time = last_lio_update_time_;
    lidar_meas.lidar = raw_cloud;
    lidar_meas.pcl_proc_cur = raw_cloud;
    lidar_meas.lio_vio_flg = LIO;
    lidar_meas.lidar_scan_index_now = 0;

    MeasureGroup meas;
    meas.lio_time = frame_end;
    meas.imu = imu_msgs;
    lidar_meas.measures.clear();
    lidar_meas.measures.push_back(meas);

    if (!first_lidar_seen_) {
      first_lidar_seen_ = true;
      first_lidar_time_ = frame_end;
      imu_process_->first_lidar_time = first_lidar_time_;
      ROS_INFO("FAST-LIVO2 recorder first lidar frame: %.6f", first_lidar_time_);
    }

    imu_process_->Process2(lidar_meas, state_, feats_undistort_);
    if (gravity_align_en_) {
      gravityAlignment();
    }

    last_lio_update_time_ = lidar_meas.last_lio_update_time;
    state_propagat_ = state_;
    voxelmap_manager_->state_ = state_;
    voxelmap_manager_->feats_undistort_ = feats_undistort_;

    if (!feats_undistort_ || feats_undistort_->empty()) {
      return false;
    }

    downsample_filter_.setInputCloud(feats_undistort_);
    downsample_filter_.filter(*feats_down_body_);
    if (!feats_down_body_ || feats_down_body_->empty()) {
      return false;
    }

    voxelmap_manager_->state_ = state_;
    voxelmap_manager_->feats_down_body_ = feats_down_body_;
    voxelmap_manager_->TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, feats_down_world_);
    voxelmap_manager_->feats_down_world_ = feats_down_world_;
    voxelmap_manager_->feats_down_size_ = static_cast<int>(feats_down_body_->size());

    if (!lidar_map_inited_) {
      lidar_map_inited_ = true;
      voxelmap_manager_->BuildVoxelMap();
    }

    voxelmap_manager_->StateEstimation(state_propagat_);
    state_ = voxelmap_manager_->state_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>());
    voxelmap_manager_->TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    const size_t n = std::min(world_lidar->points.size(), voxelmap_manager_->pv_list_.size());
    for (size_t i = 0; i < n; ++i) {
      voxelmap_manager_->pv_list_[i].point_w << world_lidar->points[i].x,
          world_lidar->points[i].y,
          world_lidar->points[i].z;
      if (i < voxelmap_manager_->cross_mat_list_.size() &&
          i < voxelmap_manager_->body_cov_list_.size()) {
        M3D point_crossmat = voxelmap_manager_->cross_mat_list_[i];
        M3D var = voxelmap_manager_->body_cov_list_[i];
        var = (state_.rot_end * ext_r_) * var * (state_.rot_end * ext_r_).transpose() +
              (-point_crossmat) * state_.cov.block<3, 3>(0, 0) *
                  (-point_crossmat).transpose() +
              state_.cov.block<3, 3>(3, 3);
        voxelmap_manager_->pv_list_[i].var = var;
      }
    }
    voxelmap_manager_->UpdateVoxelMap(voxelmap_manager_->pv_list_);
    if (voxelmap_manager_->config_setting_.map_sliding_en) {
      voxelmap_manager_->mapSliding();
    }
    return true;
  }

  void gravityAlignment()
  {
    if (!imu_process_->imu_need_init && !gravity_align_finished_) {
      V3D ez(0, 0, -1), gz(state_.gravity);
      Eigen::Quaterniond g_q_i0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
      M3D g_r_i0 = g_q_i0.toRotationMatrix();
      state_.pos_end = g_r_i0 * state_.pos_end;
      state_.rot_end = g_r_i0 * state_.rot_end;
      state_.vel_end = g_r_i0 * state_.vel_end;
      state_.gravity = g_r_i0 * state_.gravity;
      gravity_align_finished_ = true;
      ROS_INFO("FAST-LIVO2 recorder gravity alignment finished");
    }
  }

  bool lookupExternalPose(double stamp_sec, ExternalPose &pose, std::string &error)
  {
    std::string direct_error;
    try {
      geometry_msgs::TransformStamped tf_msg = tf_buffer_.lookupTransform(
          map_frame_, pose_frame_, ros::Time(stamp_sec), ros::Duration(tf_timeout_));
      fillPoseFromTransform(stamp_sec, tf_msg, pose);
      return true;
    } catch (const tf2::TransformException &exc) {
      direct_error = exc.what();
    }

    if (pose_fallback_from_extrinsic_ &&
        !pose_fallback_frame_.empty() &&
        pose_fallback_frame_ != pose_frame_) {
      try {
        geometry_msgs::TransformStamped tf_msg = tf_buffer_.lookupTransform(
            map_frame_, pose_fallback_frame_, ros::Time(stamp_sec), ros::Duration(tf_timeout_));
        fillPoseFromFallbackTransform(stamp_sec, tf_msg, pose);
        return true;
      } catch (const tf2::TransformException &fallback_exc) {
        direct_error += "; fallback " + map_frame_ + "->" + pose_fallback_frame_ +
                        ":" + fallback_exc.what();
      }
    }

    if (!allow_latest_tf_fallback_) {
      error = direct_error;
      return false;
    }

    try {
      geometry_msgs::TransformStamped tf_msg = tf_buffer_.lookupTransform(
          map_frame_, pose_frame_, ros::Time(0), ros::Duration(tf_timeout_));
      fillPoseFromTransform(stamp_sec, tf_msg, pose);
      return true;
    } catch (const tf2::TransformException &latest_exc) {
      if (!(pose_fallback_from_extrinsic_ &&
            !pose_fallback_frame_.empty() &&
            pose_fallback_frame_ != pose_frame_)) {
        error = direct_error + "; latest:" + latest_exc.what();
        return false;
      }

      try {
        geometry_msgs::TransformStamped tf_msg = tf_buffer_.lookupTransform(
            map_frame_, pose_fallback_frame_, ros::Time(0), ros::Duration(tf_timeout_));
        fillPoseFromFallbackTransform(stamp_sec, tf_msg, pose);
        return true;
      } catch (const tf2::TransformException &fallback_latest_exc) {
        error = direct_error + "; latest:" + latest_exc.what() +
                "; fallback_latest:" + fallback_latest_exc.what();
        return false;
      }
    }
  }

  void fillPoseFromTransform(double stamp_sec,
                             const geometry_msgs::TransformStamped &tf_msg,
                             ExternalPose &pose)
  {
    pose.stamp = stamp_sec;
    pose.t << tf_msg.transform.translation.x,
        tf_msg.transform.translation.y,
        tf_msg.transform.translation.z;
    pose.q = Eigen::Quaterniond(tf_msg.transform.rotation.w,
                                tf_msg.transform.rotation.x,
                                tf_msg.transform.rotation.y,
                                tf_msg.transform.rotation.z)
                 .normalized();
  }

  void fillPoseFromFallbackTransform(double stamp_sec,
                                     const geometry_msgs::TransformStamped &tf_msg,
                                     ExternalPose &pose)
  {
    Eigen::Vector3d t_map_fallback(tf_msg.transform.translation.x,
                                   tf_msg.transform.translation.y,
                                   tf_msg.transform.translation.z);
    Eigen::Quaterniond q_map_fallback(tf_msg.transform.rotation.w,
                                      tf_msg.transform.rotation.x,
                                      tf_msg.transform.rotation.y,
                                      tf_msg.transform.rotation.z);
    q_map_fallback.normalize();
    Eigen::Quaterniond q_fallback_lidar(ext_r_);
    q_fallback_lidar.normalize();

    pose.stamp = stamp_sec;
    pose.t = t_map_fallback + q_map_fallback * ext_t_;
    pose.q = (q_map_fallback * q_fallback_lidar).normalized();
  }

  bool saveFrame(double stamp_sec, const ExternalPose &pose)
  {
    fs::path pcd_path;
    std::string timestamp = formatTimestamp(stamp_sec);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!active_ || !session_pcd_dir_) {
        return false;
      }
      pcd_path = *session_pcd_dir_ / (timestamp + ".pcd");
    }

    pcl::PCDWriter writer;
    if (writer.writeBinary(pcd_path.string(), *feats_down_body_) != 0) {
      rejectFrame("pcd_write_failed:" + pcd_path.string());
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (pose_file_.is_open()) {
        pose_file_ << timestamp << " "
                   << std::fixed << std::setprecision(9)
                   << pose.t.x() << " " << pose.t.y() << " " << pose.t.z() << " "
                   << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " "
                   << pose.q.w() << "\n";
      }
      if (frames_csv_.is_open()) {
        frames_csv_ << saved_count_ << ","
                    << timestamp << ","
                    << pcd_path.filename().string() << ","
                    << feats_undistort_->size() << ","
                    << feats_down_body_->size() << ","
                    << pose.t.x() << ","
                    << pose.t.y() << ","
                    << pose.t.z() << ","
                    << pose.q.x() << ","
                    << pose.q.y() << ","
                    << pose.q.z() << ","
                    << pose.q.w() << "\n";
      }
      ++saved_count_;
      last_saved_pcd_ = pcd_path.string();
      last_reason_ = "saved:frame";
      last_error_ = "none";
      if (saved_count_ % 20 == 0) {
        writeMetadata("recording");
      }
      publishStatusLocked();
    }
    return true;
  }

  std::string formatTimestamp(double stamp_sec)
  {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << stamp_sec;
    return ss.str();
  }

  bool startRecording(std::string &message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      message = "already_recording:" + session_dir_.string();
      return false;
    }

    resetFrontend();
    cloud_count_ = 0;
    saved_count_ = 0;
    skipped_count_ = 0;
    rejected_count_ = 0;
    last_error_ = "none";
    last_reason_ = "none";
    last_saved_pcd_ = "none";

    fs::path root(expandUser(session_root_));
    fs::create_directories(root);
    const int64_t now = static_cast<int64_t>(ros::Time::now().toSec() > 0.0
                                                 ? ros::Time::now().toSec()
                                                 : ros::WallTime::now().toSec());
    fs::path candidate = root / (session_prefix_ + "_" + std::to_string(now));
    int suffix = 1;
    while (fs::exists(candidate)) {
      candidate = root / (session_prefix_ + "_" + std::to_string(now) + "_" +
                          std::to_string(suffix++));
    }
    session_dir_ = candidate;
    session_pcd_dir_ = session_dir_ / "Log" / "pcd";
    fs::create_directories(*session_pcd_dir_);

    pose_file_.open((*session_pcd_dir_ / "lidar_poses.txt").string(), std::ios::out);
    frames_csv_.open((session_dir_ / "frames.csv").string(), std::ios::out);
    frames_csv_ << "seq,timestamp,pcd,undistorted_points,downsampled_points,"
                   "x,y,z,qx,qy,qz,qw\n";

    active_ = true;
    writeMetadata("recording");
    publishStatusLocked();
    message = "session_started:" + session_dir_.string();
    return true;
  }

  bool stopRecording(std::string &message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
      message = "not_recording";
      return false;
    }
    active_ = false;
    closeSessionFiles();
    writeMetadata("stopped");
    publishStatusLocked();
    message = "session_stopped:" + session_dir_.string() +
              " saved=" + std::to_string(saved_count_) +
              " skipped=" + std::to_string(skipped_count_) +
              " rejected=" + std::to_string(rejected_count_);
    return true;
  }

  void closeSessionFiles()
  {
    if (pose_file_.is_open()) {
      pose_file_.close();
    }
    if (frames_csv_.is_open()) {
      frames_csv_.close();
    }
  }

  void writeMetadata(const std::string &state)
  {
    if (session_dir_.empty()) {
      return;
    }
    std::ofstream out((session_dir_ / "metadata.yaml").string(), std::ios::out);
    out << "format: fast_livo2_embedded_lio_frame_session\n";
    out << "state: " << state << "\n";
    out << "session_dir: " << session_dir_.string() << "\n";
    out << "pointcloud_topic: " << pointcloud_topic_ << "\n";
    out << "imu_topic: " << imu_topic_ << "\n";
    out << "map_frame: " << map_frame_ << "\n";
    out << "pose_frame: " << pose_frame_ << "\n";
    out << "pose_fallback_from_extrinsic: "
        << (pose_fallback_from_extrinsic_ ? "true" : "false") << "\n";
    out << "pose_fallback_frame: " << pose_fallback_frame_ << "\n";
    out << "pcd_frame: lidar_frame_downsampled_feats_down_body\n";
    out << "pose_semantics: map_to_pose_frame_at_lidar_frame_end\n";
    out << "saved_frames: " << saved_count_ << "\n";
    out << "skipped_clouds: " << skipped_count_ << "\n";
    out << "rejected_clouds: " << rejected_count_ << "\n";
    out << "last_error: " << last_error_ << "\n";
    out << "fast_livo2_frontend:\n";
    out << "  preprocess_cpp: embedded\n";
    out << "  imu_processing_cpp: embedded\n";
    out << "  voxel_map_cpp: embedded\n";
    out << "  filter_size_surf: " << filter_size_surf_ << "\n";
  }

  bool handleStart(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    std::string message;
    res.success = startRecording(message);
    res.message = message;
    return true;
  }

  bool handleStop(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    std::string message;
    res.success = stopRecording(message);
    res.message = message;
    return true;
  }

  bool handleStatus(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &res)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    res.success = true;
    res.message = statusTextLocked();
    return true;
  }

  void skipFrame(const std::string &reason)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++skipped_count_;
    last_reason_ = reason;
    publishStatusLocked();
  }

  void rejectFrame(const std::string &reason)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++rejected_count_;
    last_error_ = reason;
    last_reason_ = reason;
    publishStatusLocked();
  }

  void statusTimer(const ros::TimerEvent &)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
      writeMetadata("recording");
    }
    publishStatusLocked();
  }

  void publishStatusLocked()
  {
    std_msgs::String msg;
    msg.data = statusTextLocked();
    status_pub_.publish(msg);
  }

  std::string statusTextLocked() const
  {
    std::ostringstream ss;
    ss << "active=" << (active_ ? "true" : "false")
       << " session=" << (session_dir_.empty() ? "none" : session_dir_.string())
       << " cloud=" << cloud_count_
       << " saved=" << saved_count_
       << " skipped=" << skipped_count_
       << " rejected=" << rejected_count_
       << " lio_inited=" << (lidar_map_inited_ ? "true" : "false")
       << " last_saved=" << last_saved_pcd_
       << " last_reason=" << last_reason_
       << " last_error=" << last_error_;
    return ss.str();
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber imu_sub_;
  ros::Publisher status_pub_;
  ros::ServiceServer start_srv_;
  ros::ServiceServer stop_srv_;
  ros::ServiceServer status_srv_;
  ros::Timer status_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex mutex_;
  std::deque<sensor_msgs::ImuConstPtr> imu_buffer_;

  std::unique_ptr<Preprocess> preprocess_;
  std::unique_ptr<ImuProcess> imu_process_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_storage_;
  VoxelMapManagerPtr voxelmap_manager_;
  pcl::VoxelGrid<PointType> downsample_filter_;
  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr feats_down_world_;
  StatesGroup state_;
  StatesGroup state_propagat_;
  VoxelMapConfig voxel_config_;
  V3D ext_t_ = V3D::Zero();
  M3D ext_r_ = M3D::Identity();

  bool active_ = false;
  bool auto_start_ = false;
  bool imu_en_ = true;
  bool gravity_align_en_ = true;
  bool gravity_align_finished_ = false;
  bool lidar_map_inited_ = false;
  bool first_lidar_seen_ = false;
  bool allow_latest_tf_fallback_ = false;
  bool pose_fallback_from_extrinsic_ = true;

  std::string pointcloud_topic_;
  std::string imu_topic_;
  std::string session_root_;
  std::string session_prefix_;
  std::string map_frame_;
  std::string pose_frame_;
  std::string pose_fallback_frame_;
  fs::path session_dir_;
  std::optional<fs::path> session_pcd_dir_;
  std::ofstream pose_file_;
  std::ofstream frames_csv_;

  int lidar_type_ = VELO16;
  int scan_line_ = 16;
  int point_filter_num_ = 1;
  int imu_init_frame_num_ = 20;
  int min_frame_points_ = 300;
  int max_imu_buffer_size_ = 5000;

  double blind_ = 0.5;
  double filter_size_surf_ = 0.1;
  double lidar_time_offset_ = 0.0;
  double tf_timeout_ = 0.05;
  double acc_cov_ = 0.5;
  double gyr_cov_ = 0.3;
  double b_acc_cov_ = 0.0001;
  double b_gyr_cov_ = 0.0001;
  double imu_init_buffer_time_ = 2.0;
  double imu_overlap_time_ = 0.02;
  double imu_prune_keep_time_ = 1.0;
  double imu_end_tolerance_ = 0.02;
  double last_lio_update_time_ = -1.0;
  double first_lidar_time_ = 0.0;

  uint64_t cloud_count_ = 0;
  uint64_t saved_count_ = 0;
  uint64_t skipped_count_ = 0;
  uint64_t rejected_count_ = 0;
  std::string last_error_ = "none";
  std::string last_reason_ = "none";
  std::string last_saved_pcd_ = "none";
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fast_livo2_lio_recorder");
  FastLivo2LioRecorder recorder;
  ros::spin();
  return 0;
}

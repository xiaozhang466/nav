#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <costmap_2d/cost_values.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <nav_msgs/Path.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf/transform_datatypes.h>

namespace um980_driver {

namespace {

double clampValue(double value, double low, double high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

double normalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

double distance2d(const geometry_msgs::PoseStamped& first,
                  const geometry_msgs::PoseStamped& second) {
  const double dx = first.pose.position.x - second.pose.position.x;
  const double dy = first.pose.position.y - second.pose.position.y;
  return std::hypot(dx, dy);
}

double limitDelta(double previous, double target, double max_delta) {
  return previous + clampValue(target - previous, -max_delta, max_delta);
}

}  // namespace

class RegulatedPurePursuitPlanner : public nav_core::BaseLocalPlanner {
 public:
  RegulatedPurePursuitPlanner() = default;

  void initialize(std::string name,
                  tf2_ros::Buffer* tf,
                  costmap_2d::Costmap2DROS* costmap_ros) override {
    if (initialized_) {
      ROS_WARN("RegulatedPurePursuitPlanner has already been initialized");
      return;
    }
    if (tf == nullptr || costmap_ros == nullptr) {
      ROS_ERROR("RegulatedPurePursuitPlanner requires tf and costmap_ros");
      return;
    }

    tf_buffer_ = tf;
    costmap_ros_ = costmap_ros;
    global_frame_ = costmap_ros_->getGlobalFrameID();
    robot_base_frame_ = costmap_ros_->getBaseFrameID();

    ros::NodeHandle private_nh("~/" + name);
    private_nh.param("desired_linear_vel", desired_linear_vel_, 0.30);
    private_nh.param("min_linear_vel", min_linear_vel_, 0.05);
    private_nh.param("max_linear_vel", max_linear_vel_, 0.35);
    private_nh.param("min_lookahead", min_lookahead_, 1.2);
    private_nh.param("max_lookahead", max_lookahead_, 2.8);
    private_nh.param("lookahead_time", lookahead_time_, 1.5);
    private_nh.param("xy_goal_tolerance", xy_goal_tolerance_, 0.30);
    private_nh.param("yaw_goal_tolerance", yaw_goal_tolerance_, 1.57);
    private_nh.param("max_angular_vel", max_angular_vel_, 0.55);
    private_nh.param("max_linear_accel", max_linear_accel_, 0.40);
    private_nh.param("max_angular_accel", max_angular_accel_, 0.80);
    private_nh.param("angular_deadband", angular_deadband_, 0.03);
    private_nh.param("lateral_deadband", lateral_deadband_, 0.03);
    private_nh.param("curvature_slowdown_gain", curvature_slowdown_gain_, 1.2);
    private_nh.param("approach_slowdown_distance", approach_slowdown_distance_, 1.0);
    private_nh.param("transform_tolerance", transform_tolerance_, 0.2);
    private_nh.param("stop_on_obstacle", stop_on_obstacle_, true);
    private_nh.param("obstacle_check_distance", obstacle_check_distance_, 1.0);
    private_nh.param("obstacle_check_step", obstacle_check_step_, 0.10);
    private_nh.param("obstacle_cost_threshold", obstacle_cost_threshold_, 253);
    private_nh.param("obstacle_unknown_is_blocking", obstacle_unknown_is_blocking_, false);
    private_nh.param("publish_debug", publish_debug_, true);

    desired_linear_vel_ = clampValue(desired_linear_vel_, 0.0, max_linear_vel_);
    min_linear_vel_ = clampValue(min_linear_vel_, 0.0, max_linear_vel_);
    min_lookahead_ = std::max(0.05, min_lookahead_);
    max_lookahead_ = std::max(min_lookahead_, max_lookahead_);
    obstacle_check_step_ = std::max(0.02, obstacle_check_step_);

    if (publish_debug_) {
      status_pub_ = private_nh.advertise<std_msgs::String>("status", 1);
      local_plan_pub_ = private_nh.advertise<nav_msgs::Path>("local_plan", 1);
    }

    initialized_ = true;
    ROS_INFO_STREAM("RegulatedPurePursuitPlanner initialized in global frame "
                    << global_frame_ << " with base frame " << robot_base_frame_);
  }

  bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override {
    if (!initialized_) {
      ROS_ERROR("RegulatedPurePursuitPlanner has not been initialized");
      return false;
    }
    global_plan_ = plan;
    last_nearest_index_ = 0;
    goal_reached_ = false;
    return !global_plan_.empty();
  }

  bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override {
    cmd_vel = geometry_msgs::Twist();
    if (!initialized_) {
      ROS_ERROR("RegulatedPurePursuitPlanner has not been initialized");
      return false;
    }
    if (global_plan_.empty()) {
      ROS_WARN_THROTTLE(1.0, "RegulatedPurePursuitPlanner has an empty plan");
      return false;
    }

    geometry_msgs::PoseStamped robot_pose;
    if (!costmap_ros_->getRobotPose(robot_pose)) {
      ROS_WARN_THROTTLE(1.0, "RegulatedPurePursuitPlanner cannot get robot pose");
      return false;
    }

    std::vector<geometry_msgs::PoseStamped> plan;
    if (!transformPlan(plan)) {
      return false;
    }
    if (plan.size() < 2) {
      ROS_WARN_THROTTLE(1.0, "RegulatedPurePursuitPlanner plan has fewer than two poses");
      return false;
    }

    std::vector<double> cumulative;
    buildCumulativeDistances(plan, cumulative);
    const size_t nearest_index = findNearestIndex(plan, robot_pose);
    const double remaining_distance = cumulative.back() - cumulative[nearest_index];

    if (isAtGoal(robot_pose, plan.back())) {
      goal_reached_ = true;
      last_cmd_v_ = 0.0;
      last_cmd_w_ = 0.0;
      last_cmd_time_ = ros::Time::now();
      publishStatus("goal_reached", nearest_index, 0.0, 0.0, 0.0, 0.0,
                    0.0, remaining_distance);
      return true;
    }
    goal_reached_ = false;

    const double lookahead = computeLookahead();
    const size_t target_index = findLookaheadIndex(cumulative, nearest_index, lookahead);
    const geometry_msgs::PoseStamped& target_pose = plan[target_index];
    publishLocalPlan(plan, nearest_index, target_index);

    const double robot_yaw = tf::getYaw(robot_pose.pose.orientation);
    const double dx = target_pose.pose.position.x - robot_pose.pose.position.x;
    const double dy = target_pose.pose.position.y - robot_pose.pose.position.y;
    const double target_x = std::cos(robot_yaw) * dx + std::sin(robot_yaw) * dy;
    double target_y = -std::sin(robot_yaw) * dx + std::cos(robot_yaw) * dy;
    if (std::fabs(target_y) < lateral_deadband_) {
      target_y = 0.0;
    }

    const double target_distance_sq = std::max(1e-6, target_x * target_x + target_y * target_y);
    const double curvature = 2.0 * target_y / target_distance_sq;

    std::string reason = "track";
    if (stop_on_obstacle_ && pathHasObstacle(robot_pose, target_pose)) {
      reason = "obstacle";
      last_cmd_v_ = 0.0;
      last_cmd_w_ = 0.0;
      last_cmd_time_ = ros::Time::now();
      publishStatus(reason, nearest_index, lookahead, target_x, target_y,
                    curvature, 0.0, remaining_distance);
      return true;
    }

    double linear = regulatedLinearVelocity(curvature, remaining_distance);
    if (target_x < 0.0) {
      linear = std::min(linear, min_linear_vel_);
      reason = "target_behind";
    }

    double angular = linear * curvature;
    if (std::fabs(angular) > max_angular_vel_) {
      angular = clampValue(angular, -max_angular_vel_, max_angular_vel_);
      if (std::fabs(curvature) > 1e-6) {
        linear = std::min(linear, std::fabs(angular / curvature));
      }
      reason = "angular_limit";
    }
    if (std::fabs(angular) < angular_deadband_) {
      angular = 0.0;
    }

    limitAcceleration(linear, angular);
    cmd_vel.linear.x = linear;
    cmd_vel.angular.z = angular;
    publishStatus(reason, nearest_index, lookahead, target_x, target_y,
                  curvature, linear, remaining_distance);
    return true;
  }

  bool isGoalReached() override {
    return goal_reached_;
  }

 private:
  bool transformPlan(std::vector<geometry_msgs::PoseStamped>& transformed_plan) {
    transformed_plan.clear();
    transformed_plan.reserve(global_plan_.size());
    for (geometry_msgs::PoseStamped pose : global_plan_) {
      if (pose.header.frame_id.empty()) {
        pose.header.frame_id = global_frame_;
      }
      pose.header.stamp = ros::Time(0);
      geometry_msgs::PoseStamped transformed_pose;
      try {
        transformed_pose = tf_buffer_->transform(
            pose, global_frame_, ros::Duration(transform_tolerance_));
      } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(
            1.0, "RegulatedPurePursuitPlanner cannot transform plan from %s to %s: %s",
            pose.header.frame_id.c_str(), global_frame_.c_str(), ex.what());
        return false;
      }
      transformed_plan.push_back(transformed_pose);
    }
    return !transformed_plan.empty();
  }

  void buildCumulativeDistances(const std::vector<geometry_msgs::PoseStamped>& plan,
                                std::vector<double>& cumulative) const {
    cumulative.assign(plan.size(), 0.0);
    for (size_t i = 1; i < plan.size(); ++i) {
      cumulative[i] = cumulative[i - 1] + distance2d(plan[i - 1], plan[i]);
    }
  }

  size_t findNearestIndex(const std::vector<geometry_msgs::PoseStamped>& plan,
                          const geometry_msgs::PoseStamped& robot_pose) {
    if (last_nearest_index_ >= plan.size()) {
      last_nearest_index_ = 0;
    }
    const size_t search_start = std::min(last_nearest_index_, plan.size() - 1);
    double best_distance = std::numeric_limits<double>::infinity();
    size_t best_index = search_start;
    for (size_t i = search_start; i < plan.size(); ++i) {
      const double dist = distance2d(plan[i], robot_pose);
      if (dist < best_distance) {
        best_distance = dist;
        best_index = i;
      }
    }
    last_nearest_index_ = best_index;
    return best_index;
  }

  size_t findLookaheadIndex(const std::vector<double>& cumulative,
                            size_t nearest_index,
                            double lookahead) const {
    const double target_distance = cumulative[nearest_index] + lookahead;
    for (size_t i = nearest_index; i < cumulative.size(); ++i) {
      if (cumulative[i] >= target_distance) {
        return i;
      }
    }
    return cumulative.size() - 1;
  }

  double computeLookahead() const {
    return clampValue(
        min_lookahead_ + lookahead_time_ * std::fabs(last_cmd_v_),
        min_lookahead_,
        max_lookahead_);
  }

  bool isAtGoal(const geometry_msgs::PoseStamped& robot_pose,
                const geometry_msgs::PoseStamped& goal_pose) const {
    const double xy_error = distance2d(robot_pose, goal_pose);
    const double yaw_error = std::fabs(normalizeAngle(
        tf::getYaw(goal_pose.pose.orientation) - tf::getYaw(robot_pose.pose.orientation)));
    return xy_error <= xy_goal_tolerance_ && yaw_error <= yaw_goal_tolerance_;
  }

  double regulatedLinearVelocity(double curvature, double remaining_distance) const {
    const double abs_curvature = std::fabs(curvature);
    double linear = clampValue(desired_linear_vel_, min_linear_vel_, max_linear_vel_);
    linear = std::min(linear, desired_linear_vel_ / (1.0 + curvature_slowdown_gain_ * abs_curvature));

    if (abs_curvature > 1e-6) {
      linear = std::min(linear, max_angular_vel_ / abs_curvature);
    }

    if (approach_slowdown_distance_ > 1e-6 &&
        remaining_distance < approach_slowdown_distance_) {
      const double scale = clampValue(remaining_distance / approach_slowdown_distance_, 0.0, 1.0);
      linear = std::min(linear, min_linear_vel_ + scale * (desired_linear_vel_ - min_linear_vel_));
    }
    return clampValue(linear, min_linear_vel_, max_linear_vel_);
  }

  void limitAcceleration(double& linear, double& angular) {
    const ros::Time now = ros::Time::now();
    double dt = 0.1;
    if (!last_cmd_time_.isZero()) {
      dt = clampValue((now - last_cmd_time_).toSec(), 0.01, 0.5);
    }
    linear = limitDelta(last_cmd_v_, linear, max_linear_accel_ * dt);
    angular = limitDelta(last_cmd_w_, angular, max_angular_accel_ * dt);
    last_cmd_v_ = linear;
    last_cmd_w_ = angular;
    last_cmd_time_ = now;
  }

  bool pathHasObstacle(const geometry_msgs::PoseStamped& robot_pose,
                       const geometry_msgs::PoseStamped& target_pose) const {
    const costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (costmap == nullptr) {
      return false;
    }

    const double dx = target_pose.pose.position.x - robot_pose.pose.position.x;
    const double dy = target_pose.pose.position.y - robot_pose.pose.position.y;
    const double distance = std::hypot(dx, dy);
    const double check_distance = std::min(distance, obstacle_check_distance_);
    if (check_distance <= 1e-6) {
      return false;
    }

    const int steps = std::max(1, static_cast<int>(std::ceil(check_distance / obstacle_check_step_)));
    for (int i = 1; i <= steps; ++i) {
      const double sample_distance = check_distance * static_cast<double>(i) /
                                     static_cast<double>(steps);
      const double ratio = sample_distance / distance;
      const double wx = robot_pose.pose.position.x + dx * ratio;
      const double wy = robot_pose.pose.position.y + dy * ratio;
      unsigned int mx = 0;
      unsigned int my = 0;
      if (!costmap->worldToMap(wx, wy, mx, my)) {
        continue;
      }
      const unsigned char cost = costmap->getCost(mx, my);
      if (cost == costmap_2d::NO_INFORMATION && !obstacle_unknown_is_blocking_) {
        continue;
      }
      if (cost >= obstacle_cost_threshold_ || cost == costmap_2d::LETHAL_OBSTACLE ||
          cost == costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        return true;
      }
    }
    return false;
  }

  void publishStatus(const std::string& reason,
                     size_t nearest_index,
                     double lookahead,
                     double target_x,
                     double target_y,
                     double curvature,
                     double linear,
                     double remaining_distance) const {
    if (!publish_debug_ || !status_pub_) {
      return;
    }
    std_msgs::String msg;
    std::ostringstream out;
    out << "controller=rpp"
        << " reason=" << reason
        << " nearest_index=" << nearest_index
        << " lookahead=" << lookahead
        << " target_x=" << target_x
        << " target_y=" << target_y
        << " curvature=" << curvature
        << " linear=" << linear
        << " angular=" << last_cmd_w_
        << " remaining=" << remaining_distance;
    msg.data = out.str();
    status_pub_.publish(msg);
  }

  void publishLocalPlan(const std::vector<geometry_msgs::PoseStamped>& plan,
                        size_t nearest_index,
                        size_t target_index) const {
    if (!publish_debug_ || !local_plan_pub_) {
      return;
    }
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = global_frame_;
    const size_t start = std::min(nearest_index, plan.size() - 1);
    const size_t end = std::min(target_index, plan.size() - 1);
    for (size_t i = start; i <= end; ++i) {
      path.poses.push_back(plan[i]);
      if (i == end) {
        break;
      }
    }
    local_plan_pub_.publish(path);
  }

  bool initialized_ = false;
  bool goal_reached_ = false;
  costmap_2d::Costmap2DROS* costmap_ros_ = nullptr;
  tf2_ros::Buffer* tf_buffer_ = nullptr;
  std::string global_frame_;
  std::string robot_base_frame_;
  std::vector<geometry_msgs::PoseStamped> global_plan_;
  size_t last_nearest_index_ = 0;

  double desired_linear_vel_ = 0.30;
  double min_linear_vel_ = 0.05;
  double max_linear_vel_ = 0.35;
  double min_lookahead_ = 1.2;
  double max_lookahead_ = 2.8;
  double lookahead_time_ = 1.5;
  double xy_goal_tolerance_ = 0.30;
  double yaw_goal_tolerance_ = 1.57;
  double max_angular_vel_ = 0.55;
  double max_linear_accel_ = 0.40;
  double max_angular_accel_ = 0.80;
  double angular_deadband_ = 0.03;
  double lateral_deadband_ = 0.03;
  double curvature_slowdown_gain_ = 1.2;
  double approach_slowdown_distance_ = 1.0;
  double transform_tolerance_ = 0.2;
  double obstacle_check_distance_ = 1.0;
  double obstacle_check_step_ = 0.10;
  int obstacle_cost_threshold_ = 253;
  bool stop_on_obstacle_ = true;
  bool obstacle_unknown_is_blocking_ = false;
  bool publish_debug_ = true;

  ros::Time last_cmd_time_;
  double last_cmd_v_ = 0.0;
  double last_cmd_w_ = 0.0;
  ros::Publisher status_pub_;
  ros::Publisher local_plan_pub_;
};

}  // namespace um980_driver

PLUGINLIB_EXPORT_CLASS(um980_driver::RegulatedPurePursuitPlanner,
                       nav_core::BaseLocalPlanner)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_core/base_global_planner.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <tf/tf.h>

namespace um980_driver {

namespace {

std::string Trim(const std::string& input) {
  const char* whitespace = " \t\r\n";
  const auto begin = input.find_first_not_of(whitespace);
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = input.find_last_not_of(whitespace);
  return input.substr(begin, end - begin + 1);
}

bool StartsWith(const std::string& input, const std::string& prefix) {
  return input.size() >= prefix.size() &&
         input.compare(0, prefix.size(), prefix) == 0;
}

bool ParseDoubleAfterColon(const std::string& line, double* value) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return false;
  }
  std::stringstream stream(line.substr(colon + 1));
  stream >> *value;
  return !stream.fail();
}

bool ParseIntAfterColon(const std::string& line, int* value) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return false;
  }
  std::stringstream stream(line.substr(colon + 1));
  stream >> *value;
  return !stream.fail();
}

geometry_msgs::Quaternion QuaternionFromYaw(double yaw) {
  return tf::createQuaternionMsgFromYaw(yaw);
}

double Distance2D(double x0, double y0, double x1, double y1) {
  return std::hypot(x1 - x0, y1 - y0);
}

double NormalizeAngle(double angle) {
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

struct XYPoint {
  double x = 0.0;
  double y = 0.0;
};

XYPoint CubicBezierPoint(const XYPoint& p0, const XYPoint& p1,
                         const XYPoint& p2, const XYPoint& p3, double t) {
  const double one_minus_t = 1.0 - t;
  const double one_minus_t2 = one_minus_t * one_minus_t;
  const double t2 = t * t;

  XYPoint point;
  point.x = one_minus_t2 * one_minus_t * p0.x +
            3.0 * one_minus_t2 * t * p1.x +
            3.0 * one_minus_t * t2 * p2.x +
            t2 * t * p3.x;
  point.y = one_minus_t2 * one_minus_t * p0.y +
            3.0 * one_minus_t2 * t * p1.y +
            3.0 * one_minus_t * t2 * p2.y +
            t2 * t * p3.y;
  return point;
}

XYPoint CubicBezierDerivative(const XYPoint& p0, const XYPoint& p1,
                              const XYPoint& p2, const XYPoint& p3,
                              double t) {
  const double one_minus_t = 1.0 - t;
  XYPoint derivative;
  derivative.x =
      3.0 * one_minus_t * one_minus_t * (p1.x - p0.x) +
      6.0 * one_minus_t * t * (p2.x - p1.x) +
      3.0 * t * t * (p3.x - p2.x);
  derivative.y =
      3.0 * one_minus_t * one_minus_t * (p1.y - p0.y) +
      6.0 * one_minus_t * t * (p2.y - p1.y) +
      3.0 * t * t * (p3.y - p2.y);
  return derivative;
}

}  // namespace

struct RoutePoint {
  int id = -1;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double yaw = 0.0;
  double distance = 0.0;
  bool has_distance = false;
  bool has_yaw = false;
};

struct Projection {
  double progress = 0.0;
  double distance_error = std::numeric_limits<double>::infinity();
  double x = 0.0;
  double y = 0.0;
};

RoutePoint HermiteInterpolate(const RoutePoint& p0, const RoutePoint& p1,
                              const RoutePoint& p2, const RoutePoint& p3,
                              double t, double tangent_scale) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + t;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;

  const double m1x = tangent_scale * (p2.x - p0.x);
  const double m1y = tangent_scale * (p2.y - p0.y);
  const double m1z = tangent_scale * (p2.z - p0.z);
  const double m2x = tangent_scale * (p3.x - p1.x);
  const double m2y = tangent_scale * (p3.y - p1.y);
  const double m2z = tangent_scale * (p3.z - p1.z);

  RoutePoint output;
  output.id = -1;
  output.x = h00 * p1.x + h10 * m1x + h01 * p2.x + h11 * m2x;
  output.y = h00 * p1.y + h10 * m1y + h01 * p2.y + h11 * m2y;
  output.z = h00 * p1.z + h10 * m1z + h01 * p2.z + h11 * m2z;
  return output;
}

class RouteGlobalPlanner : public nav_core::BaseGlobalPlanner {
 public:
  RouteGlobalPlanner() = default;
  RouteGlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
    initialize(name, costmap_ros);
  }

  void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) override {
    if (initialized_) {
      return;
    }

    ros::NodeHandle private_nh("~/" + name);
    private_nh.param<std::string>("waypoints_file", waypoints_file_, waypoints_file_);
    private_nh.param("interpolation_resolution", interpolation_resolution_, 0.20);
    private_nh.param("max_projection_error", max_projection_error_, 5.0);
    private_nh.param("allow_reverse_plan", allow_reverse_plan_, false);
    private_nh.param("smooth_route", smooth_route_, true);
    private_nh.param("spline_tangent_scale", spline_tangent_scale_, 0.35);
    private_nh.param("soft_merge_enabled", soft_merge_enabled_, true);
    private_nh.param("soft_merge_min_error", soft_merge_min_error_, 0.35);
    private_nh.param("soft_merge_distance_gain", soft_merge_distance_gain_, 2.0);
    private_nh.param("soft_merge_min_entry_distance", soft_merge_min_entry_distance_, 1.5);
    private_nh.param("soft_merge_max_entry_distance", soft_merge_max_entry_distance_, 6.0);
    private_nh.param("soft_merge_heading_distance_gain", soft_merge_heading_distance_gain_, 1.8);
    private_nh.param("soft_merge_handle_scale", soft_merge_handle_scale_, 0.6);
    private_nh.param("soft_merge_max_handle_distance", soft_merge_max_handle_distance_, 3.5);

    if (costmap_ros != nullptr) {
      global_frame_ = costmap_ros->getGlobalFrameID();
    }
    private_nh.param<std::string>("global_frame", global_frame_, global_frame_);

    if (!LoadWaypoints(waypoints_file_)) {
      ROS_ERROR_STREAM("RouteGlobalPlanner failed to load waypoints file: " << waypoints_file_);
      route_.clear();
    } else {
      ROS_INFO_STREAM("RouteGlobalPlanner loaded " << route_.size()
                                                   << " route points from "
                                                   << waypoints_file_);
    }

    initialized_ = true;
  }

  bool makePlan(const geometry_msgs::PoseStamped& start,
                const geometry_msgs::PoseStamped& goal,
                std::vector<geometry_msgs::PoseStamped>& plan) override {
    plan.clear();
    if (!initialized_) {
      ROS_ERROR("RouteGlobalPlanner has not been initialized");
      return false;
    }
    if (route_.size() < 2) {
      ROS_ERROR("RouteGlobalPlanner has fewer than two route points");
      return false;
    }

    const Projection start_projection =
        ProjectToRoute(start.pose.position.x, start.pose.position.y);
    const Projection goal_projection =
        ProjectToRoute(goal.pose.position.x, goal.pose.position.y);

    if (start_projection.distance_error > max_projection_error_) {
      ROS_WARN_THROTTLE(2.0,
                        "RouteGlobalPlanner start is %.2fm from route, plan may begin with a cross-track correction",
                        start_projection.distance_error);
    }
    if (goal_projection.distance_error > max_projection_error_) {
      ROS_WARN_THROTTLE(2.0,
                        "RouteGlobalPlanner goal is %.2fm from route; projecting it onto the route",
                        goal_projection.distance_error);
    }

    double start_progress = start_projection.progress;
    double goal_progress = goal_projection.progress;
    double route_plan_start_progress = start_progress;
    const double start_yaw = tf::getYaw(start.pose.orientation);

    if (!allow_reverse_plan_ && goal_progress < start_progress) {
      ROS_WARN_THROTTLE(
          2.0,
          "RouteGlobalPlanner goal projection is behind start projection; clamping goal to start progress");
      goal_progress = start_progress;
    }

    const bool planning_forward = goal_progress >= start_progress;
    const int route_direction = planning_forward ? 1 : -1;
    const RoutePoint route_start_point = RoutePoseAt(start_progress);
    const double route_start_yaw =
        route_direction > 0 ? route_start_point.yaw : NormalizeAngle(route_start_point.yaw + M_PI);
    const double start_heading_error =
        std::fabs(NormalizeAngle(route_start_yaw - start_yaw));

    geometry_msgs::PoseStamped start_pose = start;
    start_pose.header.frame_id = global_frame_;
    start_pose.header.stamp = ros::Time(0);
    plan.push_back(start_pose);

    if (soft_merge_enabled_ &&
        start_projection.distance_error >= soft_merge_min_error_ &&
        std::fabs(goal_progress - start_progress) >= 1e-3) {
      const double entry_distance = std::min(
          soft_merge_max_entry_distance_,
          std::max(soft_merge_min_entry_distance_,
                   std::max(start_projection.distance_error * soft_merge_distance_gain_,
                            soft_merge_min_entry_distance_ +
                                start_heading_error *
                                    soft_merge_heading_distance_gain_)));
      const double direction = planning_forward ? 1.0 : -1.0;
      const double requested_progress = start_progress + direction * entry_distance;
      const double progress_lower = std::min(start_progress, goal_progress);
      const double progress_upper = std::max(start_progress, goal_progress);
      const double merge_progress =
          std::min(progress_upper, std::max(progress_lower, requested_progress));

      if (std::fabs(merge_progress - start_progress) >= 1e-3) {
        AppendSoftMergeSegment(
            start_pose, start_progress, merge_progress, start_yaw, route_direction, plan);
        route_plan_start_progress = merge_progress;
      }
    }

    const double step = std::max(0.05, interpolation_resolution_);
    if (goal_progress >= route_plan_start_progress) {
      AppendRoutePose(route_plan_start_progress, route_direction, plan);
      for (double progress = route_plan_start_progress + step; progress < goal_progress;
           progress += step) {
        AppendRoutePose(progress, route_direction, plan);
      }
    } else {
      AppendRoutePose(route_plan_start_progress, route_direction, plan);
      for (double progress = route_plan_start_progress - step; progress > goal_progress;
           progress -= step) {
        AppendRoutePose(progress, route_direction, plan);
      }
    }
    AppendRoutePose(goal_progress, route_direction, plan);

    ROS_DEBUG_STREAM("RouteGlobalPlanner produced plan with " << plan.size()
                                                              << " poses from progress "
                                                              << route_plan_start_progress << " to "
                                                              << goal_progress);
    return plan.size() >= 2;
  }

 private:
  bool LoadWaypoints(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      ROS_ERROR_STREAM("Cannot open route waypoints file: " << path);
      return false;
    }

    std::vector<RoutePoint> points;
    RoutePoint current;
    bool in_waypoint = false;
    bool has_x = false;
    bool has_y = false;

    auto flush_current = [&]() {
      if (in_waypoint && has_x && has_y) {
        points.push_back(current);
      }
      current = RoutePoint();
      in_waypoint = false;
      has_x = false;
      has_y = false;
    };

    std::string line;
    while (std::getline(file, line)) {
      std::string trimmed = Trim(line);
      if (trimmed.empty() || StartsWith(trimmed, "#")) {
        continue;
      }

      if (StartsWith(trimmed, "- id:")) {
        flush_current();
        in_waypoint = true;
        ParseIntAfterColon(trimmed, &current.id);
        continue;
      }

      if (!in_waypoint) {
        continue;
      }

      double value = 0.0;
      if (StartsWith(trimmed, "x:") && ParseDoubleAfterColon(trimmed, &value)) {
        current.x = value;
        has_x = true;
      } else if (StartsWith(trimmed, "y:") && ParseDoubleAfterColon(trimmed, &value)) {
        current.y = value;
        has_y = true;
      } else if (StartsWith(trimmed, "z:") && ParseDoubleAfterColon(trimmed, &value)) {
        current.z = value;
      } else if (StartsWith(trimmed, "yaw:") && ParseDoubleAfterColon(trimmed, &value)) {
        current.yaw = value;
        current.has_yaw = true;
      } else if (StartsWith(trimmed, "distance:") &&
                 ParseDoubleAfterColon(trimmed, &value)) {
        current.distance = value;
        current.has_distance = true;
      }
    }
    flush_current();

    if (points.size() < 2) {
      return false;
    }

    PopulateDistanceAndYaw(&points);
    route_ = smooth_route_ ? BuildSmoothedRoute(points) : points;
    return true;
  }

  void PopulateDistanceAndYaw(std::vector<RoutePoint>* points) const {
    if (points == nullptr || points->empty()) {
      return;
    }

    double cumulative = 0.0;
    (*points)[0].distance = 0.0;
    for (std::size_t index = 1; index < points->size(); ++index) {
      cumulative += Distance2D((*points)[index - 1].x, (*points)[index - 1].y,
                               (*points)[index].x, (*points)[index].y);
      (*points)[index].distance = cumulative;
    }

    if (points->size() == 1) {
      (*points)[0].yaw = 0.0;
      return;
    }

    for (std::size_t index = 0; index < points->size(); ++index) {
      if (index + 1 < points->size()) {
        (*points)[index].yaw =
            std::atan2((*points)[index + 1].y - (*points)[index].y,
                       (*points)[index + 1].x - (*points)[index].x);
      } else {
        (*points)[index].yaw = (*points)[index - 1].yaw;
      }
    }
  }

  std::vector<RoutePoint> BuildSmoothedRoute(const std::vector<RoutePoint>& control_points) const {
    if (control_points.size() < 3) {
      return control_points;
    }

    std::vector<RoutePoint> dense_route;
    dense_route.reserve(control_points.size() * 4);
    dense_route.push_back(control_points.front());

    const double step = std::max(0.05, interpolation_resolution_);
    for (std::size_t index = 0; index + 1 < control_points.size(); ++index) {
      const RoutePoint& p0 = control_points[index == 0 ? 0 : index - 1];
      const RoutePoint& p1 = control_points[index];
      const RoutePoint& p2 = control_points[index + 1];
      const RoutePoint& p3 =
          control_points[index + 2 < control_points.size() ? index + 2 : control_points.size() - 1];

      const double segment_length = Distance2D(p1.x, p1.y, p2.x, p2.y);
      const int steps = std::max(1, static_cast<int>(std::ceil(segment_length / step)));
      for (int sample = 1; sample <= steps; ++sample) {
        const double t = static_cast<double>(sample) / static_cast<double>(steps);
        RoutePoint point = HermiteInterpolate(p0, p1, p2, p3, t, spline_tangent_scale_);
        if (Distance2D(point.x, point.y, dense_route.back().x, dense_route.back().y) < 1e-4) {
          continue;
        }
        dense_route.push_back(point);
      }
    }

    PopulateDistanceAndYaw(&dense_route);
    ROS_INFO_STREAM("RouteGlobalPlanner smoothed " << control_points.size()
                                                   << " control points into "
                                                   << dense_route.size()
                                                   << " route samples");
    return dense_route;
  }

  Projection ProjectToRoute(double x, double y) const {
    Projection best;
    if (route_.empty()) {
      return best;
    }

    if (route_.size() == 1) {
      best.progress = route_[0].distance;
      best.x = route_[0].x;
      best.y = route_[0].y;
      best.distance_error = Distance2D(x, y, best.x, best.y);
      return best;
    }

    for (std::size_t index = 0; index + 1 < route_.size(); ++index) {
      const RoutePoint& first = route_[index];
      const RoutePoint& second = route_[index + 1];
      const double dx = second.x - first.x;
      const double dy = second.y - first.y;
      const double segment_sq = dx * dx + dy * dy;
      double ratio = 0.0;
      if (segment_sq > 1e-9) {
        ratio = ((x - first.x) * dx + (y - first.y) * dy) / segment_sq;
        ratio = std::max(0.0, std::min(1.0, ratio));
      }
      const double projected_x = first.x + dx * ratio;
      const double projected_y = first.y + dy * ratio;
      const double error = Distance2D(x, y, projected_x, projected_y);
      if (error < best.distance_error) {
        best.distance_error = error;
        best.x = projected_x;
        best.y = projected_y;
        best.progress =
            first.distance + (second.distance - first.distance) * ratio;
      }
    }

    return best;
  }

  RoutePoint RoutePoseAt(double progress) const {
    if (progress <= route_.front().distance) {
      return route_.front();
    }
    if (progress >= route_.back().distance) {
      return route_.back();
    }

    for (std::size_t index = 0; index + 1 < route_.size(); ++index) {
      const RoutePoint& first = route_[index];
      const RoutePoint& second = route_[index + 1];
      if (second.distance < progress) {
        continue;
      }
      const double span = std::max(1e-9, second.distance - first.distance);
      const double ratio = (progress - first.distance) / span;
      RoutePoint output;
      output.id = -1;
      output.distance = progress;
      output.x = first.x + (second.x - first.x) * ratio;
      output.y = first.y + (second.y - first.y) * ratio;
      output.z = first.z + (second.z - first.z) * ratio;
      output.yaw = std::atan2(second.y - first.y, second.x - first.x);
      return output;
    }

    return route_.back();
  }

  void AppendRoutePose(double progress, int route_direction,
                       std::vector<geometry_msgs::PoseStamped>& plan) const {
    const RoutePoint point = RoutePoseAt(progress);
    const double yaw =
        route_direction >= 0 ? point.yaw : NormalizeAngle(point.yaw + M_PI);
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = global_frame_;
    pose.header.stamp = ros::Time(0);
    pose.pose.position.x = point.x;
    pose.pose.position.y = point.y;
    pose.pose.position.z = 0.0;
    pose.pose.orientation = QuaternionFromYaw(yaw);
    plan.push_back(pose);
  }

  void AppendSoftMergeSegment(const geometry_msgs::PoseStamped& start_pose,
                              double route_start_progress,
                              double route_entry_progress,
                              double start_yaw,
                              int route_direction,
                              std::vector<geometry_msgs::PoseStamped>& plan) const {
    const RoutePoint entry_point = RoutePoseAt(route_entry_progress);
    const double merge_distance =
        Distance2D(start_pose.pose.position.x, start_pose.pose.position.y,
                   entry_point.x, entry_point.y);
    if (merge_distance < 0.5 * interpolation_resolution_) {
      return;
    }

    const double entry_handle_distance = std::min(
        soft_merge_max_handle_distance_,
        std::max(0.6, std::fabs(route_entry_progress - route_start_progress) *
                          soft_merge_handle_scale_));
    const double start_handle_distance = std::min(
        soft_merge_max_handle_distance_,
        std::max(0.6, merge_distance * 0.45));
    const double route_yaw =
        route_direction >= 0 ? entry_point.yaw : NormalizeAngle(entry_point.yaw + M_PI);
    const double route_direction_x = std::cos(route_yaw);
    const double route_direction_y = std::sin(route_yaw);
    const double start_direction_x = std::cos(start_yaw);
    const double start_direction_y = std::sin(start_yaw);

    const XYPoint p0{start_pose.pose.position.x, start_pose.pose.position.y};
    const XYPoint p1{p0.x + start_direction_x * start_handle_distance,
                     p0.y + start_direction_y * start_handle_distance};
    const XYPoint p3{entry_point.x, entry_point.y};
    const XYPoint p2{p3.x - route_direction_x * entry_handle_distance,
                     p3.y - route_direction_y * entry_handle_distance};

    const int steps =
        std::max(2, static_cast<int>(std::ceil(merge_distance /
                                               std::max(0.05, interpolation_resolution_))));
    for (int step_index = 1; step_index < steps; ++step_index) {
      const double t = static_cast<double>(step_index) /
                       static_cast<double>(steps);
      const XYPoint point = CubicBezierPoint(p0, p1, p2, p3, t);
      const XYPoint tangent = CubicBezierDerivative(p0, p1, p2, p3, t);
      if (!plan.empty() &&
          Distance2D(plan.back().pose.position.x, plan.back().pose.position.y,
                     point.x, point.y) < 1e-3) {
        continue;
      }

      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = global_frame_;
      pose.header.stamp = ros::Time(0);
      pose.pose.position.x = point.x;
      pose.pose.position.y = point.y;
      pose.pose.position.z = 0.0;
      const double yaw =
          std::hypot(tangent.x, tangent.y) > 1e-4 ? std::atan2(tangent.y, tangent.x)
                                                  : route_yaw;
      pose.pose.orientation = QuaternionFromYaw(yaw);
      plan.push_back(pose);
    }
  }

  bool initialized_ = false;
  std::string waypoints_file_;
  std::string global_frame_ = "map";
  double interpolation_resolution_ = 0.20;
  double max_projection_error_ = 5.0;
  bool allow_reverse_plan_ = false;
  bool smooth_route_ = true;
  double spline_tangent_scale_ = 0.35;
  bool soft_merge_enabled_ = true;
  double soft_merge_min_error_ = 0.35;
  double soft_merge_distance_gain_ = 2.0;
  double soft_merge_min_entry_distance_ = 1.5;
  double soft_merge_max_entry_distance_ = 6.0;
  double soft_merge_heading_distance_gain_ = 1.8;
  double soft_merge_handle_scale_ = 0.6;
  double soft_merge_max_handle_distance_ = 3.5;
  std::vector<RoutePoint> route_;
};

}  // namespace um980_driver

PLUGINLIB_EXPORT_CLASS(um980_driver::RouteGlobalPlanner, nav_core::BaseGlobalPlanner)

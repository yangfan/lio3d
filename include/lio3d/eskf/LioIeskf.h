#pragma once

#include "ieskf.h"
#include "matching/NDT_INC.h"
#include "tools/ImuInitializer.h"
#include "tools/MapViewer.h"
#include "tools/Sync.h"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <string>

class LioIeskf {
public:
  struct Params {
    double min_kf_dist = 1.0;
    double min_kf_deg = 10.0;
    bool viewer_on = true;
  };
  bool config(const std::string &yaml_file);
  void add_imu(std::unique_ptr<sensor_msgs::msg::Imu> imu_msg);
  void add_scan(std::unique_ptr<sensor_msgs::msg::PointCloud2> scan_msg);

  void save_map(const std::string &map_file);

private:
  ImuInitializer imu_initializer_;
  Sync lidar_imu_sync_;
  NDT_INC ndt_inc_;
  bool ndt_initialized = false;
  IESKF ieskf_;
  std::unique_ptr<MapViewer> viewer_;

  Sync::DataGroup lidar_imu_;
  std::vector<IMUState> states_;
  Params params_;
  double deg2rad = M_PI / 180.0;

  Sophus::SE3d T_IL_;
  Sophus::SE3d last_kf_pose_;

  void process_sync_data(const Sync::DataGroup &lidar_imu);

  bool initialize_imu(const Sync::DataGroup &lidar_imu);

  void predict();
  void undistort();
  void correct();

  bool is_keyframe(const Sophus::SE3d &pose) const;

  Sophus::SE3d interpolation(const double ratio, const IMUState &state0,
                             const IMUState &state1) const;
  Sophus::SE3d integrate_imu(const IMUState &state, const IMUPtr &imu_measure,
                             const double dt) const;
  pcl::PointCloud<pcl::PointXYZI>::Ptr desampling(const double leaf_sz);

  template <typename S>
  Eigen::Matrix<S, 3, 1> VecFromArray(const std::vector<S> &v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
  }

  template <typename S>
  Eigen::Matrix<S, 3, 3> MatFromArray(const std::vector<S> &v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
  }
};
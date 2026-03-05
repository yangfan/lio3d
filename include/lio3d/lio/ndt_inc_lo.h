#pragma once

#include "matching/NDT_INC.h"
#include "tools/MapViewer.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

class NDT_INC_LO {
public:
  using Point = pcl::PointXYZI;
  using PointCloud = pcl::PointCloud<Point>;
  struct Params {
    double min_kf_dist = 1.0;
    double min_kf_rot_rad = 10 * M_PI / 180.0;
    bool viwer_on = false;
    NDT_INC::Params ndt_params;
  };

  NDT_INC_LO() = default;
  NDT_INC_LO(const Params params) : params_(params) {
    if (params_.viwer_on) {
      set_viewer(0.5);
    }

    ndt_inc_matcher_.set_params(params_.ndt_params);
  }

  bool add_scan(PointCloud::Ptr scan);
  bool add_scan(PointCloud::Ptr scan, Sophus::SE3d &estimated_pose);

  void set_viewer(const float sz) {
    viewer_ = std::make_unique<MapViewer>("NDT incremental LO", sz);
  }

  bool save_map(const std::string &file);

  bool initialized() const { return initialized_; }

  size_t kf_num() const { return kf_num_; }

  void set_viewer_dir(const Eigen::Vector3d &dir) {
    viewer_->set_forward_dir(dir);
  }

private:
  bool initialized_ = false;

  Sophus::SE3d last_keyframe_pose_;

  Sophus::SE3d last_frame_pose_;
  Sophus::SE3d last_motion_;

  NDT_INC ndt_inc_matcher_;

  std::unique_ptr<MapViewer> viewer_;
  Params params_;

  size_t kf_num_ = 0;

  bool is_keyframe(const Sophus::SE3d &cur_pose) const;
};
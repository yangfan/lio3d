#pragma once

#include "matching/NDT.h"
#include "tools/MapViewer.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.hpp>

#include <deque>

class NDT_LO {
public:
  using Point = pcl::PointXYZI;
  using PointCloud = pcl::PointCloud<Point>;
  struct Params {
    double min_kf_dist = 0.5;
    double min_kf_rot_rad = 30 * M_PI / 180.0;
    size_t max_kf_num = 30;
    bool viwer_on = false;
    NDT::Params ndt_params;
  };
  struct Frame {
    Frame() = default;
    Frame(const size_t i, PointCloud::Ptr s, const Sophus::SE3d &pose)
        : id(i), scan(s), Twf(pose) {}
    size_t id;
    PointCloud::Ptr scan;
    Sophus::SE3d Twf;
  };

  NDT_LO() = default;
  NDT_LO(const Params params) : params_(params) {
    if (params_.viwer_on) {
      set_viewer(0.5);
    }

    ndt_matcher_.set_params(params_.ndt_params);
  }

  bool add_scan(PointCloud::Ptr scan);
  bool is_keyframe(const Frame &frame) const;

  void set_viewer(const float sz) {
    viewer_ = std::make_unique<MapViewer>("NDT LO", sz);
  }

  bool save_map(const std::string &file);

  void set_viewer_dir(const Eigen::Vector3d &dir) {
    viewer_->set_forward_dir(dir);
  }

private:
  PointCloud::Ptr map_;
  std::deque<std::unique_ptr<Frame>> keyframes_;

  Sophus::SE3d last_frame_pose_;
  Sophus::SE3d last_motion_;

  NDT ndt_matcher_;
  std::unique_ptr<MapViewer> viewer_;
  Params params_;
};
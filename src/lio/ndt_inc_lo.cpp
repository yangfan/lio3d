#include "lio/ndt_inc_lo.h"

#include <glog/logging.h>

bool NDT_INC_LO::add_scan(PointCloud::Ptr scan) {
  if (scan->empty()) {
    return false;
  }
  if (!initialized_) {
    LOG(INFO) << "Initializing.";
    ndt_inc_matcher_.add_scan(scan);
    last_keyframe_pose_ = Sophus::SE3d();
    last_frame_pose_ = last_keyframe_pose_;
    if (viewer_) {
      viewer_->add_pointcloud(scan, Sophus::SE3d());
    }
    initialized_ = true;
    LOG(INFO) << "Initialized.";
    return true;
  }
  Sophus::SE3d estimated_pose = last_frame_pose_ * last_motion_;
  if (!ndt_inc_matcher_.align(estimated_pose, scan)) {
    return false;
  }
  LOG(INFO) << "Estimated pose: " << estimated_pose.translation().transpose();
  last_motion_ = last_frame_pose_.inverse() * estimated_pose;
  last_frame_pose_ = estimated_pose;

  auto aligned_scan = std::make_shared<PointCloud>();
  pcl::transformPointCloud(*scan, *aligned_scan,
                           estimated_pose.matrix().cast<float>());

  if (is_keyframe(estimated_pose)) {
    last_keyframe_pose_ = estimated_pose;
    ndt_inc_matcher_.add_scan(aligned_scan);
  }
  if (viewer_) {
    viewer_->add_pointcloud(aligned_scan, estimated_pose);
  }
  return true;
}

bool NDT_INC_LO::is_keyframe(const Sophus::SE3d &cur_pose) const {
  const Sophus::SE3d rel_motion = last_keyframe_pose_.inverse() * cur_pose;
  return rel_motion.translation().norm() > params_.min_kf_dist ||
         rel_motion.so3().log().norm() > params_.min_kf_rot_rad;
}

bool NDT_INC_LO::save_map(const std::string &file) {
  if (!viewer_) {
    return false;
  }
  return viewer_->save_map(file);
}

bool NDT_INC_LO::add_scan(PointCloud::Ptr scan, Sophus::SE3d &estimated_pose) {
  if (scan->empty()) {
    return false;
  }
  if (!initialized_) {
    LOG(INFO) << "Initializing.";
    ndt_inc_matcher_.add_scan(scan);
    if (viewer_) {
      viewer_->add_pointcloud(scan, Sophus::SE3d());
    }
    initialized_ = true;
    LOG(INFO) << "Initialized.";
    return true;
  }
  Sophus::SE3d guess = estimated_pose;
  if (!ndt_inc_matcher_.align(guess, scan)) {
    return false;
  }

  auto aligned_scan = std::make_shared<PointCloud>();
  pcl::transformPointCloud(*scan, *aligned_scan, guess.matrix().cast<float>());

  estimated_pose = guess;

  if (is_keyframe(estimated_pose)) {
    ndt_inc_matcher_.add_scan(aligned_scan);
    kf_num_++;
  }
  if (viewer_) {
    viewer_->add_pointcloud(aligned_scan, estimated_pose);
  }
  return true;
}
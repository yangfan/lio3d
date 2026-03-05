#include "lio/ndt_lo.h"

#include <glog/logging.h>

bool NDT_LO::add_scan(PointCloud::Ptr scan) {
  if (!map_) {
    map_ = scan;
    keyframes_.emplace_back(std::make_unique<Frame>(0, scan, Sophus::SE3d()));
    last_frame_pose_ = keyframes_.back()->Twf;
    last_motion_ = Sophus::SE3d();
    if (viewer_) {
      viewer_->add_pointcloud(scan, Sophus::SE3d());
    }
    ndt_matcher_.set_target_cloud(map_);
    return true;
  }
  ndt_matcher_.set_source_cloud(scan);
  Sophus::SE3d estimated_pose = last_frame_pose_ * last_motion_;

  if (!ndt_matcher_.align(estimated_pose)) {
    return false;
  }
  auto aligned_scan = std::make_shared<PointCloud>();
  pcl::transformPointCloud(*scan, *aligned_scan,
                           estimated_pose.matrix().cast<float>());
  auto cur_frame = std::make_unique<Frame>(keyframes_.back()->id + 1,
                                           aligned_scan, estimated_pose);

  if (viewer_) {
    viewer_->add_pointcloud(cur_frame->scan, cur_frame->Twf);
  }

  last_motion_ = last_frame_pose_.inverse() * cur_frame->Twf;
  last_frame_pose_ = cur_frame->Twf;

  if (is_keyframe(*cur_frame)) {
    keyframes_.emplace_back(std::move(cur_frame));
    if (keyframes_.size() > params_.max_kf_num) {
      keyframes_.pop_front();
    }
    map_.reset(new PointCloud);
    for (const auto &kf : keyframes_) {
      *map_ += *(kf->scan);
    }
    ndt_matcher_.set_target_cloud(map_);
  }
  return true;
}

bool NDT_LO::is_keyframe(const Frame &frame) const {
  if (keyframes_.empty()) {
    return true;
  }
  const Sophus::SE3d rel_motion = keyframes_.back()->Twf.inverse() * frame.Twf;
  return rel_motion.translation().norm() > params_.min_kf_dist ||
         rel_motion.so3().log().norm() > params_.min_kf_rot_rad;
}

bool NDT_LO::save_map(const std::string &file) {
  if (!viewer_) {
    pcl::io::savePCDFile(file, *map_);
    return false;
  }
  viewer_->save_map(file);
  LOG(INFO) << "Saved map at " << file;
  LOG(INFO) << "Close viewer to stop program.";
  viewer_->spin();
  return true;
}
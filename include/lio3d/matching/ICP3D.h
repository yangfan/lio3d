#pragma once

#include "KDTree.hpp"

#include <Eigen/Core>
#include <sophus/se3.hpp>

using PointCloud3D = KDTree<double, 3>::PointCloud;

class ICP3D {
public:
  using Mat6 = Eigen::Matrix<double, 6, 6>;
  using Vec6 = Eigen::Matrix<double, 6, 1>;
  struct Params {
    Params() = default;
    Params(const int it, const double max_d, const int min_v)
        : iterations(it), max_dist(max_d), min_valid(min_v) {}
    int iterations = 10;
    double max_dist = 1.0;
    int min_valid = 10;
    double eps = 1e-3;
  };

  ICP3D() = default;
  explicit ICP3D(const Params params) : param_(params) {}

  void set_target_cloud(PointCloud3D &&pcl_cloud) {
    target_cloud_ = std::move(pcl_cloud);
    kd_tree_.setInputCloud(target_cloud_);
    t_center_ = std::accumulate(target_cloud_.begin(), target_cloud_.end(),
                                Eigen::Vector3d::Zero().eval()) /
                target_cloud_.size();
  }
  void set_source_cloud(PointCloud3D &&pcl_cloud) {
    source_cloud_ = std::move(pcl_cloud);
    s_center_ = std::accumulate(source_cloud_.begin(), source_cloud_.end(),
                                Eigen::Vector3d::Zero().eval()) /
                source_cloud_.size();
  }

  bool align_p2p(Sophus::SE3d &Tts);
  bool align_p2l(Sophus::SE3d &Tts);
  bool align_p2pl(Sophus::SE3d &Tts);

private:
  KDTree<double, 3> kd_tree_;
  PointCloud3D target_cloud_;
  Eigen::Vector3d t_center_ = Eigen::Vector3d::Zero();
  PointCloud3D source_cloud_;
  Eigen::Vector3d s_center_ = Eigen::Vector3d::Zero();
  Params param_;

  bool fitting_line(const std::vector<int> &tidx, Eigen::Vector3d &p0,
                    Eigen::Vector3d &d);
  bool fitting_plane(const std::vector<int> &tidx, Eigen::Vector3d &n,
                     double &d);
};
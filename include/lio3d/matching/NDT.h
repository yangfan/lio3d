#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

class NDT {
public:
  using VoxelId = Eigen::Vector3i;
  using PointCloud3D =
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
  using Mat36 = Eigen::Matrix<double, 3, 6>;
  using Mat6 = Eigen::Matrix<double, 6, 6>;
  using Vec6 = Eigen::Matrix<double, 6, 1>;
  using Vec3 = Eigen::Matrix<double, 3, 1>;
  struct Voxel {
    Voxel() = default;
    Voxel(const size_t tid) : pids({tid}) {}
    std::vector<size_t> pids;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
  };
  // Optimized Spatial Hashing for Collision Detection of Deformable Objects,
  // 2003
  struct hash_pt3 {
    size_t operator()(const Eigen::Vector3i &pt) const {
      return size_t((pt[0] * 73856093) ^ (pt[1] * 19349663) ^
                    (pt[2] * 83492791) % 10000000);
    }
  };

  enum class NeighborType { NB0, NB6, NB14, NB27 };

  struct Params {
    int iterations = 20;
    int min_valid = 10;
    double eps = 1e-3;
    NeighborType nb_type = NeighborType::NB6;
    double vx_size = 1.0; // unit: meter
    size_t min_vx_pt = 3;
    double max_dist = 1.0;
    double chi2_th = 20.0;
  };

  void set_params(const Params &param) { params_ = param; }
  void set_neighbors(const NeighborType type);
  bool set_target_cloud(PointCloud3D &&cloud);
  bool set_source_cloud(PointCloud3D &&cloud);

  bool nearest_neighbors(const Eigen::Vector3d &query_pt, const size_t k,
                         std::vector<int> &nearest_idx,
                         std::vector<double> &nearest_dist);
  bool nearest_neighbors_kmt(const PointCloud3D &query_pc, const size_t k,
                             std::vector<std::vector<int>> &nearest_idx,
                             std::vector<std::vector<double>> &nearest_dist);
  bool align(Sophus::SE3d &Tts);

  VoxelId get_id(const Eigen::Vector3d &p) const {
    return (p / params_.vx_size).array().round().cast<int>();
  }

  size_t size() const { return grid_.size(); }
  size_t point_num() const;

  const std::unordered_map<VoxelId, Voxel, hash_pt3> &grid() { return grid_; }

private:
  std::unordered_map<VoxelId, Voxel, hash_pt3> grid_;

  Params params_;

  PointCloud3D target_cloud_;
  PointCloud3D source_cloud_;
  Eigen::Vector3d target_center_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d source_center_ = Eigen::Vector3d::Zero();

  std::vector<VoxelId, Eigen::aligned_allocator<VoxelId>> neighbors_;

  bool build_grid();
  bool mean_cov(const Voxel &voxel, Eigen::Vector3d &mean,
                Eigen::Matrix3d &cov) const;
};
#include "ICP3D.h"

#include <Eigen/Dense>

#include <execution>

bool ICP3D::align_p2p(Sophus::SE3d &Tts) {
  if (target_cloud_.empty() || source_cloud_.empty()) {
    return false;
  }

  Sophus::SE3d pose = Tts;
  pose.translation() = t_center_ - s_center_;
  double last_err = std::numeric_limits<double>::max();

  const int sz = source_cloud_.size();

  std::vector<Eigen::Matrix<double, 3, 6>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>>
      Js(sz);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> Es(
      sz);

  PointCloud3D source_transformed(sz);
  std::vector<bool> valid(sz, true);

  std::vector<size_t> idx(sz);
  std::iota(idx.begin(), idx.end(), 0);

  for (int i = 0; i < param_.iterations; ++i) {

    int valid_cnt = 0;
    double sq_err = 0.0;

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [this, &source_transformed, &pose](const size_t sid) {
                    source_transformed[sid] = pose * source_cloud_[sid];
                  });
    std::vector<std::vector<int>> nearest_idx;
    std::vector<std::vector<double>> nearest_dist;
    nearest_idx.reserve(source_cloud_.size());
    nearest_dist.reserve(source_cloud_.size());
    kd_tree_.nearest_neighbors_kmt(source_transformed, 1, nearest_idx,
                                   nearest_dist);

    std::for_each(
        std::execution::par_unseq, idx.begin(), idx.end(),
        [this, &pose, &Js, &Es, &valid, &valid_cnt, &sq_err, &nearest_idx,
         &nearest_dist, &source_transformed](const size_t sid) {
          if (nearest_dist[sid][0] > param_.max_dist) {
            valid[sid] = false;
            return;
          }
          Js[sid].block<3, 3>(0, 0) =
              pose.so3().matrix() * Sophus::SO3d::hat(source_cloud_[sid]);
          Js[sid].block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
          Es[sid] =
              target_cloud_[nearest_idx[sid][0]] - source_transformed[sid];
          sq_err += Es[sid].squaredNorm();
          valid_cnt++;
          valid[sid] = true;
        });

    if (valid_cnt < param_.min_valid) {
      return false;
    }
    double avg_err = sq_err / int(valid_cnt);
    LOG(INFO) << "It " << i << " sq err: " << sq_err
              << ", valid cnt: " << valid_cnt << ", avg err: " << avg_err;
    // if (avg_err > last_err) {
    //   break;
    // }
    last_err = avg_err;

    auto S = std::accumulate(
        idx.begin(), idx.end(),
        std::pair<Mat6, Vec6>(Mat6::Zero(), Vec6::Zero()),
        [&valid, &Js, &Es](const std::pair<Mat6, Vec6> &sum,
                           size_t sid) -> std::pair<Mat6, Vec6> {
          if (valid[sid]) {
            return std::pair<Mat6, Vec6>(
                sum.first + Js[sid].transpose() * Js[sid],
                sum.second - Js[sid].transpose() * Es[sid]);
          }
          return sum;
        });

    const Eigen::Matrix<double, 6, 1> delta = S.first.ldlt().solve(S.second);
    if (std::isnan(delta[0]) || delta.norm() < param_.eps) {
      break;
    }
    pose.translation() = pose.translation() + delta.tail<3>();
    pose.so3() = pose.so3() * Sophus::SO3d::exp(delta.head<3>());
  }
  Tts = pose;

  return true;
}

bool ICP3D::align_p2l(Sophus::SE3d &Tts) {
  if (target_cloud_.empty() || source_cloud_.empty()) {
    return false;
  }
  Sophus::SE3d pose = Tts;
  pose.translation() = t_center_ - s_center_;
  const size_t sz = source_cloud_.size();

  std::vector<size_t> idx(sz);
  std::iota(idx.begin(), idx.end(), 0);

  using Mat36 = Eigen::Matrix<double, 3, 6>;
  std::vector<Mat36, Eigen::aligned_allocator<Mat36>> Js(sz);
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> es(
      sz);
  std::vector<bool> valid(sz, true);

  PointCloud3D source_transformed(sz);

  double last_err = std::numeric_limits<double>::max();

  for (int i = 0; i < param_.iterations; ++i) {

    double sq_err = 0.0;
    int valid_cnt = 0;

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [this, &source_transformed, &pose](const size_t sid) {
                    source_transformed[sid] = pose * source_cloud_[sid];
                  });
    std::vector<std::vector<int>> nearest_idx;
    std::vector<std::vector<double>> nearest_dist;
    nearest_idx.reserve(sz);
    nearest_dist.reserve(sz);
    kd_tree_.nearest_neighbors_kmt(source_transformed, param_.fitting_num,
                                   nearest_idx, nearest_dist);

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [this, &pose, &valid, &nearest_idx, &es, &Js, &sq_err,
                   &valid_cnt, &source_transformed](const int sid) {
                    if (nearest_idx.size() < param_.fitting_num) {
                      valid[sid] = false;
                      return;
                    }
                    Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
                    Eigen::Vector3d d = Eigen::Vector3d::Zero();
                    if (!fitting_line(nearest_idx[sid], p0, d)) {
                      valid[sid] = false;
                      return;
                    }
                    Eigen::Matrix3d dhat = Sophus::SO3d::hat(d);
                    es[sid] = dhat * (source_transformed[sid] - p0);
                    if (es[sid].norm() > 0.5) {
                      valid[sid] = false;
                      return;
                    }
                    Js[sid].block<3, 3>(0, 0) =
                        -dhat * (pose.so3().matrix() *
                                 Sophus::SO3d::hat(source_cloud_[sid]));
                    Js[sid].block<3, 3>(0, 3) = dhat;

                    sq_err += es[sid].squaredNorm();
                    valid_cnt++;
                    valid[sid] = true;
                  });

    if (valid_cnt < param_.min_valid) {
      return false;
    }
    const double avg_err = sq_err / valid_cnt;
    LOG(INFO) << "It " << i << " sq err: " << sq_err
              << ", valid cnt: " << valid_cnt << ", avg err: " << avg_err;
    if (avg_err > last_err) {
      LOG(INFO) << "last err: " << last_err;
      break;
    }
    last_err = avg_err;

    using linear_eq = std::pair<Mat6, Vec6>;
    auto Hb = std::accumulate(
        idx.begin(), idx.end(), linear_eq(Mat6::Zero(), Vec6::Zero()),
        [&valid, &Js, &es](const linear_eq &sum,
                           const size_t sid) -> linear_eq {
          if (valid[sid]) {
            return linear_eq(sum.first + Js[sid].transpose() * Js[sid],
                             sum.second - Js[sid].transpose() * es[sid]);
          }
          return sum;
        });
    const Vec6 delta = Hb.first.ldlt().solve(Hb.second);
    if (std::isnan(delta[0]) || delta.norm() < param_.eps) {
      LOG(INFO) << "delta norm: " << delta.norm();
      break;
    }
    pose.so3() = pose.so3() * Sophus::SO3d::exp(delta.head<3>());
    pose.translation() = pose.translation() + delta.tail<3>();
  }
  Tts = pose;

  return true;
}

bool ICP3D::fitting_line(const std::vector<int> &tidx, Eigen::Vector3d &p0,
                         Eigen::Vector3d &d) {
  if (tidx.size() < 3) {
    return false;
  }
  p0 = std::accumulate(tidx.begin(), tidx.end(), Eigen::Vector3d::Zero().eval(),
                       [this](const Eigen::Vector3d &sum,
                              const int tid) -> Eigen::Vector3d {
                         return sum + target_cloud_[tid];
                       }) /
       tidx.size();
  Eigen::MatrixX<double> A(tidx.size(), 3);
  for (size_t i = 0; i < tidx.size(); ++i) {
    A.row(i) = (target_cloud_[tidx[i]] - p0).transpose();
  }
  Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
  d = svd.matrixV().col(0);

  return true;
}

bool ICP3D::align_p2pl(Sophus::SE3d &Tts) {
  if (target_cloud_.empty() || source_cloud_.empty()) {
    return false;
  }
  const size_t sz = source_cloud_.size();
  Sophus::SE3d pose = Tts;
  pose.translation() = t_center_ - s_center_;

  PointCloud3D source_transformed(sz);
  std::vector<int> idx(sz);
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<std::vector<int>> nearest_idx(sz);
  std::vector<std::vector<double>> nearest_dist(sz);
  std::vector<bool> valid(sz, true);
  std::vector<Eigen::Matrix<double, 1, 6>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 1, 6>>>
      Js(sz);
  std::vector<double> es(sz);

  double last_err = std::numeric_limits<double>::max();

  for (int i = 0; i < param_.iterations; ++i) {

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [&pose, &source_transformed, this](const size_t sid) {
                    source_transformed[sid] = pose * source_cloud_[sid];
                  });
    kd_tree_.nearest_neighbors_kmt(source_transformed, param_.fitting_num,
                                   nearest_idx, nearest_dist);
    double sq_err = 0.0;
    int valid_cnt = 0;
    std::for_each(
        std::execution::par_unseq, idx.begin(), idx.end(),
        [&sq_err, &valid_cnt, &nearest_idx, &valid, &source_transformed, &Js,
         &es, &pose, this](const size_t sid) {
          if (nearest_idx[sid].size() < param_.fitting_num) {
            valid[sid] = false;
            return;
          }
          Eigen::Vector3d n = Eigen::Vector3d::Zero();
          double d = 0.0;
          if (!fitting_plane(nearest_idx[sid], n, d)) {
            valid[sid] = false;
            return;
          }
          es[sid] = n.transpose() * source_transformed[sid] + d;
          if (es[sid] > 0.05) {
            valid[sid] = false;
            return;
          }
          Js[sid].head<3>() =
              -n.transpose() *
              (pose.so3().matrix() * Sophus::SO3d::hat(source_cloud_[sid]));
          Js[sid].tail<3>() = n.transpose();

          sq_err += es[sid] * es[sid];
          valid_cnt++;
          valid[sid] = true;
        });
    if (valid_cnt < param_.min_valid) {
      return false;
    }
    const double avg_err = sq_err / valid_cnt;
    LOG(INFO) << "It " << i << " sq err: " << sq_err
              << ", valid cnt: " << valid_cnt << ", avg err: " << avg_err;
    if (avg_err > last_err) {
      LOG(INFO) << "last err: " << last_err;
      break;
    }
    last_err = avg_err;

    using linear_eq = std::pair<Mat6, Vec6>;
    auto Hb = std::accumulate(
        idx.begin(), idx.end(), linear_eq(Mat6::Zero(), Vec6::Zero()),
        [&valid, &Js, &es](const linear_eq &sum,
                           const size_t sid) -> linear_eq {
          if (valid[sid]) {
            return linear_eq(sum.first + Js[sid].transpose() * Js[sid],
                             sum.second - Js[sid].transpose() * es[sid]);
          }
          return sum;
        });
    const Eigen::Matrix<double, 6, 1> delta = Hb.first.ldlt().solve(Hb.second);
    if (std::isnan(delta[0]) || delta.norm() < param_.eps) {
      break;
    }
    pose.so3() = pose.so3() * Sophus::SO3d::exp(delta.head<3>());
    pose.translation() = pose.translation() + delta.tail<3>();
  }
  Tts = pose;

  return true;
}

bool ICP3D::fitting_plane(const std::vector<int> &tidx, Eigen::Vector3d &n,
                          double &d) {
  if (tidx.size() < 4) {
    return false;
  }
  Eigen::MatrixX<double> A(tidx.size(), 4);
  A.setOnes();
  for (size_t tid = 0; tid < tidx.size(); ++tid) {
    A.row(tid).head<3>() = target_cloud_[tidx[tid]].transpose();
  }
  Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
  const Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
  n = x.head<3>();
  d = x[3];

  return true;
}
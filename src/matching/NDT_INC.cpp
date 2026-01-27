#include "NDT_INC.h"

#include <execution>
#include <set>

bool NDT_INC::add_scan(PointCloudPtr scan) {
  if (scan->empty()) {
    return false;
  }

  std::set<Voxel *> vptrs;

  for (const auto &point : scan->points) {
    const Vec3 pt = pos(point);
    const VoxelId vid = get_id(pt);
    auto vit = grid_.find(vid);

    if (vit == grid_.end()) {
      data_.push_front(std::pair(vid, std::make_unique<Voxel>(pt)));
      grid_.insert({vid, data_.begin()});

      if (grid_.size() > params_.grid_capacity) {
        grid_.erase(data_.back().first);
        data_.pop_back();
      }
    } else {

      Voxel *voxel = vit->second->second.get();
      if (voxel->pts.size() < params_.max_vx_pt) {
        voxel->pts.emplace_back(pt);
      }
      data_.splice(data_.begin(), data_, vit->second);
    }
    vptrs.insert(grid_[vid]->second.get());
  }
  LOG(INFO) << "Evaluating voxels.";

  std::for_each(std::execution::par_unseq, vptrs.begin(), vptrs.end(),
                [this](Voxel *vptr) { evaluate_voxel(vptr); });

  initial_scan_inserted = true;
  return true;
}

void NDT_INC::evaluate_voxel(Voxel *voxel) {

  if (!initial_scan_inserted) {
    if (voxel->pts.size() > 1) {
      mean_cov(*voxel, voxel->mean, voxel->cov);
      // voxel->info = info_mat(voxel->cov);
      voxel->info = (voxel->cov + Eigen::Matrix3d::Identity() * 1e-3).inverse();

    } else {
      voxel->mean = voxel->pts[0];
      voxel->cov = Eigen::Matrix3d::Identity() * 1e-2;
      voxel->info = Eigen::Matrix3d::Identity() * 1e2;
    }
    voxel->initial_evaluated = true;
    voxel->num_evaluated_pts += voxel->pts.size();
    voxel->pts.clear();
    return;
  }
  if (voxel->initial_evaluated &&
      voxel->num_evaluated_pts > params_.max_vx_pt) {
    return;
  }
  // pts.size(): number of pts in voxel but not evaluated yet
  if (!voxel->initial_evaluated && voxel->pts.size() > params_.min_vx_pt) {
    mean_cov(*voxel, voxel->mean, voxel->cov);
    // voxel->info = info_mat(voxel->cov);
    voxel->info = (voxel->cov + Eigen::Matrix3d::Identity() * 1e-3).inverse();
    voxel->initial_evaluated = true;

  } else if (voxel->initial_evaluated &&
             voxel->pts.size() > params_.min_vx_pt) {
    Eigen::Matrix3d new_cov;
    Eigen::Vector3d new_mean;
    mean_cov(*voxel, new_mean, new_cov);

    update_mean_cov(*voxel, new_mean, new_cov);
    voxel->info = info_mat(voxel->cov);
  }

  voxel->num_evaluated_pts += voxel->pts.size();
  voxel->pts.clear();
  return;
}

bool NDT_INC::align(Sophus::SE3d &Tts, PointCloudPtr source) {
  if (grid_.empty() || source->empty()) {
    return false;
  }
  if (neighbors_.empty()) {
    set_neighbors(params_.nb_type);
  }
  const size_t sz = source->size();
  size_t nb_num = neighbors_.size();
  const size_t edge_sz = sz * nb_num;

  Sophus::SE3d pose = Tts;

  std::vector<size_t> idx(sz);
  std::iota(idx.begin(), idx.end(), 0);

  std::vector<Mat36, Eigen::aligned_allocator<Mat36>> Js(edge_sz);
  std::vector<Vec3, Eigen::aligned_allocator<Vec3>> es(edge_sz);
  std::vector<Voxel *> vptrs(edge_sz, nullptr);
  std::vector<bool> valid(edge_sz, false);

  double last_chi2 = std::numeric_limits<double>::max();
  int valid_cnt = 0;

  for (int i = 0; i < params_.iterations; ++i) {
    double cur_chi2 = 0.0;
    valid_cnt = 0;

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                  [this, &source, &Js, &es, &valid, &vptrs, &cur_chi2,
                   &valid_cnt, &nb_num, &pose](const size_t sid) {
                    const Eigen::Vector3d query_pt =
                        pose * pos(source->points[sid]);
                    const VoxelId cur_id = get_id(query_pt);

                    for (size_t nid = 0; nid < nb_num; ++nid) {

                      const auto &nb = neighbors_[nid];
                      auto nb_it = grid_.find(cur_id + nb);
                      const size_t eid = sid * nb_num + nid;

                      if (nb_it == grid_.end() ||
                          !nb_it->second->second->initial_evaluated) {
                        valid[eid] = false;
                        continue;
                      }
                      const Voxel &voxel = *(nb_it->second->second);
                      es[eid] = query_pt - voxel.mean;
                      const double chi2 =
                          es[eid].transpose() * voxel.info * es[eid];

                      if (std::isnan(chi2) || chi2 > params_.chi2_th) {
                        valid[eid] = false;
                        continue;
                      }
                      Js[eid].block<3, 3>(0, 0) =
                          -pose.so3().matrix() *
                          Sophus::SO3d::hat(pos(source->points[sid]));
                      Js[eid].block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
                      vptrs[eid] = nb_it->second->second.get();
                      valid[eid] = true;

                      cur_chi2 += chi2;
                      valid_cnt++;
                    }
                  });
    if (valid_cnt < params_.min_valid) {
      return false;
    }
    const double avg_chi2 = cur_chi2 / valid_cnt;
    LOG(INFO) << "It " << i << " cur_chi2: " << cur_chi2
              << ", valid_cnt: " << valid_cnt << ", avg_chi2: " << avg_chi2;
    if (avg_chi2 > 1.2 * last_chi2) {
      LOG(INFO) << "Ths solution is getting worse. Stop optimization.";
      break;
    }
    last_chi2 = avg_chi2;

    Mat6 H = Mat6::Zero();
    Vec6 b = Vec6::Zero();
    for (size_t eid = 0; eid < valid.size(); ++eid) {
      if (valid[eid]) {
        H += Js[eid].transpose() * vptrs[eid]->info * Js[eid];
        b += -Js[eid].transpose() * vptrs[eid]->info * es[eid];
      }
    }
    Vec6 delta = H.ldlt().solve(b);
    if (std::isnan(delta[0]) || delta.norm() < params_.eps) {
      LOG(INFO) << "coverged, delta: " << delta.transpose();
      break;
    }
    pose.so3() = pose.so3() * Sophus::SO3d::exp(delta.head<3>());
    pose.translation() = pose.translation() + delta.tail<3>();
  }
  Tts = pose;

  return true;
}

bool NDT_INC::mean_cov(const Voxel &voxel, Eigen::Vector3d &mean,
                       Eigen::Matrix3d &cov) const {
  mean = std::accumulate(voxel.pts.begin(), voxel.pts.end(),
                         Eigen::Vector3d::Zero().eval()) /
         voxel.pts.size();
  cov = std::accumulate(voxel.pts.begin(), voxel.pts.end(),
                        Eigen::Matrix3d::Zero().eval(),
                        [&mean](const Eigen::Matrix3d &sum,
                                const Vec3 pt) -> Eigen::Matrix3d {
                          const Eigen::Vector3d diff = pt - mean;
                          return sum + diff * diff.transpose();
                        }) /
        (voxel.pts.size() - 1);
  return true;
}

void NDT_INC::update_mean_cov(Voxel &voxel, const Eigen::Vector3d &new_mean,
                              const Eigen::Matrix3d &new_cov) {
  const size_t old_sz = voxel.num_evaluated_pts;
  const size_t new_sz = voxel.pts.size();

  const Eigen::Vector3d updated_mean =
      (old_sz * voxel.mean + new_sz * new_mean) / (old_sz + new_sz);

  const Eigen::Vector3d diff_old = voxel.mean - updated_mean;
  const Eigen::Vector3d diff_new = new_mean - updated_mean;

  const Eigen::Matrix3d updated_cov =
      (old_sz * (voxel.cov + diff_old * diff_old.transpose()) +
       new_sz * (new_cov + diff_new * diff_new.transpose())) /
      (old_sz + new_sz);
  voxel.mean = updated_mean;
  voxel.cov = updated_cov;
  return;
}

Eigen::Matrix3d NDT_INC::info_mat(const Eigen::Matrix3d &cov) {

  // svd inverse: good for voxel with more points
  // entries in info mat would be too large if too few points
  // in that case, (cov + identiy * small).inverse perform better
  // compare with method above, svd inverse is closer to matrix inverse
  Eigen::JacobiSVD svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singulars = svd.singularValues();
  singulars[1] = std::max(singulars[1], singulars[0] * 1e-3);
  singulars[2] = std::max(singulars[2], singulars[0] * 1e-3);
  Eigen::Matrix3d singulars_inv =
      Eigen::Vector3d(1.0 / singulars[0], 1.0 / singulars[1],
                      1.0 / singulars[2])
          .asDiagonal();
  return svd.matrixV() * singulars_inv * svd.matrixU().transpose();
}

void NDT_INC::set_neighbors(const NeighborType type) {
  if (type == NeighborType::NB0) {
    neighbors_ = {VoxelId(0, 0, 0)};
  } else if (type == NeighborType::NB6) {
    neighbors_ = {VoxelId(0, 0, 0), VoxelId(1, 0, 0),  VoxelId(-1, 0, 0),
                  VoxelId(0, 1, 0), VoxelId(0, -1, 0), VoxelId(0, 0, 1),
                  VoxelId(0, 0, -1)};
  } else if (type == NeighborType::NB14) {
    neighbors_ = {
        VoxelId(0, 0, 0),   VoxelId(1, 0, 0),   VoxelId(-1, 0, 0),
        VoxelId(0, 1, 0),   VoxelId(0, -1, 0),  VoxelId(0, 0, 1),
        VoxelId(0, 0, -1),  VoxelId(1, 1, 1),   VoxelId(1, 1, -1),
        VoxelId(1, -1, 1),  VoxelId(-1, 1, 1),  VoxelId(-1, -1, 1),
        VoxelId(1, -1, -1), VoxelId(-1, 1, -1),
    };
  }
}
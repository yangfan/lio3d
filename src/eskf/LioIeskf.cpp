#include "eskf/LioIeskf.h"

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <execution>

bool LioIeskf::config(const std::string &yaml_file) {

  lidar_imu_sync_ = Sync([this](const Sync::DataGroup &lidar_imu) {
    process_sync_data(lidar_imu);
  });
  lidar_imu_sync_.config(yaml_file);

  imu_initializer_.config(yaml_file);

  if (params_.viewer_on) {
    viewer_ = std::make_unique<MapViewer>("IESKF LIO", 0.5);
  }

  NDT_INC::Params ndt_params;
  ndt_params.nb_type = NDT_INC::NeighborType::NB6;
  ndt_params.vx_size = 1.0;
  ndt_params.min_vx_pt = 5;
  ndt_params.iterations = 5;
  ndt_params.chi2_th = 5.0;
  ndt_params.guess_translation = false;
  ndt_inc_.set_params(ndt_params);

  auto yaml = YAML::LoadFile(yaml_file);
  std::vector<double> ext_t =
      yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
  std::vector<double> ext_r =
      yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

  const Eigen::Vector3d lidar_T_wrt_IMU = VecFromArray(ext_t);
  const Eigen::Matrix3d lidar_R_wrt_IMU = MatFromArray(ext_r);
  T_IL_ = Sophus::SE3d(lidar_R_wrt_IMU, lidar_T_wrt_IMU);

  return true;
}

void LioIeskf::add_imu(std::unique_ptr<sensor_msgs::msg::Imu> imu_msg) {
  lidar_imu_sync_.add_imu(std::move(imu_msg));
}

void LioIeskf::add_scan(
    std::unique_ptr<sensor_msgs::msg::PointCloud2> scan_msg) {
  lidar_imu_sync_.add_cloud(std::move(scan_msg));
}

void LioIeskf::process_sync_data(const Sync::DataGroup &lidar_imu) {
  lidar_imu_ = lidar_imu;
  if (!imu_initializer_.success()) {
    initialize_imu(lidar_imu_);
    return;
  }

  predict();

  undistort();

  correct();
}

bool LioIeskf::initialize_imu(const Sync::DataGroup &lidar_imu) {
  for (const auto &imu : lidar_imu.imu_sequence) {
    if (imu_initializer_.addImu(*imu)) {
      break;
    }
  }
  if (imu_initializer_.success()) {
    ieskf_.initialize_noise(imu_initializer_);
    ieskf_.initialize_pose(Sophus::SE3d(), imu_initializer_.timestamp());
  }
  return ieskf_.pose_initialized() && ieskf_.noise_initialized();
}

void LioIeskf::predict() {
  states_.clear();
  states_.reserve(lidar_imu_.imu_sequence.size() + 1);
  states_.emplace_back(ieskf_.state());

  for (const auto &imu_data : lidar_imu_.imu_sequence) {
    ieskf_.predict_imu(*imu_data);
    states_.emplace_back(ieskf_.state());
  }

  return;
}

// p_li to p_le
void LioIeskf::undistort() {
  auto &pts = lidar_imu_.scan->points;
  std::sort(pts.begin(), pts.end(),
            [](const LidarPointType &pt_a, const LidarPointType &pt_b) {
              return pt_a.time < pt_b.time;
            });

  const double last_imu_time = states_.back().timestamp;
  const Sophus::SE3d T_W_Ie(states_.back().rot, states_.back().pos);

  size_t sid_end = 1;

  for (auto &pt : pts) {
    const double ptime = lidar_imu_.scan_start_time + pt.time * 1e-3;
    Sophus::SE3d T_W_Ii;

    if (ptime > last_imu_time) {
      T_W_Ii = integrate_imu(states_.back(), lidar_imu_.imu_sequence.back(),
                             ptime - last_imu_time);
      // T_W_Ii = T_W_Ie;

    } else {

      while (states_[sid_end].timestamp < ptime) {
        sid_end++;
      }

      const double dt =
          states_[sid_end].timestamp - states_[sid_end - 1].timestamp;
      if (dt < 1e-6) {
        LOG(WARNING) << "time window is too small: " << dt;
        T_W_Ii = Sophus::SE3d(states_[sid_end].rot, states_[sid_end].pos);

      } else {
        const double ratio = (ptime - states_[sid_end - 1].timestamp) / dt;
        T_W_Ii = interpolation(ratio, states_[sid_end - 1], states_[sid_end]);
      }
    }

    const Eigen::Vector3d p_Ie =
        T_W_Ie.inverse() * T_W_Ii * T_IL_ * pt.getVector3fMap().cast<double>();
    pt.x = float(p_Ie.x());
    pt.y = float(p_Ie.y());
    pt.z = float(p_Ie.z());
  }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr
LioIeskf::desampling(const double leaf_sz) {

  LidarPointCloudPtr cloud_I = lidar_imu_.scan;

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl_cloud->points.resize(cloud_I->size());
  std::vector<size_t> idx(cloud_I->size());
  std::iota(idx.begin(), idx.end(), 0);
  std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
                [&pcl_cloud, &cloud_I](const size_t pid) {
                  pcl_cloud->points[pid].x = cloud_I->points[pid].x;
                  pcl_cloud->points[pid].y = cloud_I->points[pid].y;
                  pcl_cloud->points[pid].z = cloud_I->points[pid].z;
                  pcl_cloud->points[pid].intensity = 0;
                });
  pcl_cloud->height = 1;
  pcl_cloud->width = pcl_cloud->size();

  pcl::VoxelGrid<pcl::PointXYZI> vg;
  vg.setLeafSize(leaf_sz, leaf_sz, leaf_sz);
  vg.setInputCloud(pcl_cloud);
  pcl::PointCloud<pcl::PointXYZI>::Ptr desmapled_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  vg.filter(*desmapled_cloud);

  return desmapled_cloud;
}

void LioIeskf::correct() {

  auto pcl_cloud = desampling(0.5);

  if (!ndt_initialized) {
    ndt_inc_.add_scan(pcl_cloud);
    last_kf_pose_ = Sophus::SE3d();

    if (viewer_) {
      viewer_->add_pointcloud(pcl_cloud, Sophus::SE3d());
    }
    ndt_initialized = true;
    return;
  }

  LOG(INFO) << "before correction: "
            << ieskf_.state_SE3().translation().transpose();
  ndt_inc_.set_source(pcl_cloud);
  ieskf_.correct_pose(
      [this](const Sophus::SE3d &pose, IESKF::Mat18 &H, IESKF::Vec18 &b) {
        ndt_inc_.compute_Hb(pose, H, b);
      });
  const Sophus::SE3d cur_pose = ieskf_.state_SE3();
  LOG(INFO) << "after correction: " << cur_pose.translation().transpose();

  if (is_keyframe(cur_pose)) {
    auto cloud_W = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::transformPointCloud(*pcl_cloud, *cloud_W,
                             cur_pose.matrix().cast<float>());
    LOG(INFO) << "Creating new KeyFrame at "
              << cur_pose.translation().transpose();
    ndt_inc_.add_scan(cloud_W);

    if (viewer_) {
      viewer_->add_pointcloud(cloud_W, cur_pose);
    }
    last_kf_pose_ = cur_pose;
  }
}

bool LioIeskf::is_keyframe(const Sophus::SE3d &pose) const {
  const Sophus::SE3d rel_motion = last_kf_pose_.inverse() * pose;
  return (rel_motion.translation().norm() > params_.min_kf_dist) ||
         (rel_motion.so3().log().norm() > params_.min_kf_deg * deg2rad);
}

Sophus::SE3d LioIeskf::integrate_imu(const IMUState &state,
                                     const IMUPtr &imu_measure,
                                     const double dt) const {
  const Eigen::Vector3d last_acc = imu_measure->acc;
  const Eigen::Vector3d last_omega = imu_measure->gyr;

  const Eigen::Vector3d pos =
      state.pos + state.vel * dt + 0.5 * state.gravity * dt * dt +
      0.5 * (state.rot * (last_acc - state.bias_a)) * dt * dt;
  const Sophus::SO3d rot =
      state.rot * Sophus::SO3d::exp((last_omega - state.bias_g) * dt);
  return Sophus::SE3d(rot, pos);
}

Sophus::SE3d LioIeskf::interpolation(const double ratio, const IMUState &state0,
                                     const IMUState &state1) const {
  const Sophus::SE3d pose0(state0.rot, state0.pos);
  const Sophus::SE3d pose1(state1.rot, state1.pos);

  const Eigen::Vector3d pos =
      (1 - ratio) * pose0.translation() + ratio * pose1.translation();
  const Sophus::SO3d rot = Sophus::SO3d(
      pose0.unit_quaternion().slerp(ratio, pose1.unit_quaternion()));

  return Sophus::SE3d(rot, pos);
}

void LioIeskf::save_map(const std::string &map_file) {
  if (viewer_) {
    viewer_->save_map(map_file);
  }
}
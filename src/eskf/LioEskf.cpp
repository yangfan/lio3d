#include "eskf/LioEskf.h"

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include <execution>

bool LioEskf::config(const std::string &yaml_file) {

  lidar_imu_sync_ = Sync([this](const Sync::DataGroup &lidar_imu) {
    process_sync_data(lidar_imu);
  });
  lidar_imu_sync_.config(yaml_file);

  imu_initializer_.config(yaml_file);

  params_.save_undistortion = false;

  NDT_INC_LO::Params params;
  params.viwer_on = true;
  params.ndt_params.nb_type = NDT_INC::NeighborType::NB6;
  params.ndt_params.vx_size = 1.0;
  params.ndt_params.min_vx_pt = 5;
  params.ndt_params.iterations = 5;
  params.ndt_params.chi2_th = 5.0;
  params.ndt_params.guess_translation = false;

  ndt_lo_ = NDT_INC_LO(params);

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

void LioEskf::add_imu(std::unique_ptr<sensor_msgs::msg::Imu> imu_msg) {
  lidar_imu_sync_.add_imu(std::move(imu_msg));
}

void LioEskf::add_scan(
    std::unique_ptr<sensor_msgs::msg::PointCloud2> scan_msg) {
  lidar_imu_sync_.add_cloud(std::move(scan_msg));
}

void LioEskf::process_sync_data(const Sync::DataGroup &lidar_imu) {
  lidar_imu_ = lidar_imu;
  if (!imu_initializer_.success()) {
    initialize_imu(lidar_imu_);
    return;
  }

  predict();

  undistort();

  correct();
}

bool LioEskf::initialize_imu(const Sync::DataGroup &lidar_imu) {
  for (const auto &imu : lidar_imu.imu_sequence) {
    if (imu_initializer_.addImu(*imu)) {
      break;
    }
  }
  if (imu_initializer_.success()) {
    eskf_.initialize_noise(imu_initializer_);
    eskf_.initialize_pose(Sophus::SE3d(), imu_initializer_.timestamp());
  }
  return eskf_.pose_initialized() && eskf_.noise_initialized();
}

void LioEskf::predict() {
  states_.clear();
  states_.reserve(lidar_imu_.imu_sequence.size() + 1);
  states_.emplace_back(eskf_.state());

  for (const auto &imu_data : lidar_imu_.imu_sequence) {
    eskf_.predict_imu(*imu_data);
    states_.emplace_back(eskf_.state());
  }

  return;
}

// p_li to p_le
void LioEskf::undistort() {
  auto &pts = lidar_imu_.scan->points;
  std::sort(pts.begin(), pts.end(),
            [](const LidarPointType &pt_a, const LidarPointType &pt_b) {
              return pt_a.time < pt_b.time;
            });

  const double last_imu_time = states_.back().timestamp;
  const Sophus::SE3d T_W_Ie(states_.back().rot, states_.back().pos);

  size_t sid_end = 1;

  //   if (params_.save_undistortion && states_.back().vel.norm() > 1.0) {
  //     lidar_imu_.scan->height = 1;
  //     lidar_imu_.scan->width = lidar_imu_.scan->size();
  //     pcl::io::savePCDFile(
  //         "/home/fan/ssd/Projects/ros2_ws/src/lio3d/data/output/before.pcd",
  //         *lidar_imu_.scan);
  //   }

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

    // const Eigen::Vector3d p_Le = T_IL_.inverse() * T_W_Ie.inverse() * T_W_Ii
    // *
    //                              T_IL_ * pt.getVector3fMap().cast<double>();
    // pt.x = float(p_Le.x());
    // pt.y = float(p_Le.y());
    // pt.z = float(p_Le.z());
    const Eigen::Vector3d p_Ie =
        T_W_Ie.inverse() * T_W_Ii * T_IL_ * pt.getVector3fMap().cast<double>();
    pt.x = float(p_Ie.x());
    pt.y = float(p_Ie.y());
    pt.z = float(p_Ie.z());
  }

  //   if (params_.save_undistortion && states_.back().vel.norm() > 1.0) {
  //     lidar_imu_.scan->height = 1;
  //     lidar_imu_.scan->width = lidar_imu_.scan->size();
  //     pcl::io::savePCDFile(
  //         "/home/fan/ssd/Projects/ros2_ws/src/lio3d/data/output/after.pcd",
  //         *lidar_imu_.scan);
  //     params_.save_undistortion = false;
  //   }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr LioEskf::desampling(const double leaf_sz) {

  // // p_Le -> p_Ie
  // LidarPointCloudPtr cloud_I(new LidarPointCloud);
  // pcl::transformPointCloud(*lidar_imu_.scan, *cloud_I,
  //                          T_IL_.matrix().cast<float>());
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

void LioEskf::correct() {

  auto pcl_cloud = desampling(0.5);
  Sophus::SE3d T_W_Ie = Sophus::SE3d(states_.back().rot, states_.back().pos);
  if (!ndt_lo_.initialized()) {
    T_W_Ie = Sophus::SE3d();
    ndt_lo_.add_scan(pcl_cloud, T_W_Ie);

    return;
  }
  ndt_lo_.add_scan(pcl_cloud, T_W_Ie);

  LOG(INFO) << "before correction: " << T_W_Ie.translation().transpose();
  eskf_.correct_pose(T_W_Ie, states_.back().timestamp);
  LOG(INFO) << "after correction: " << states_.back().pos.transpose();
}

Sophus::SE3d LioEskf::integrate_imu(const IMUState &state,
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

Sophus::SE3d LioEskf::interpolation(const double ratio, const IMUState &state0,
                                    const IMUState &state1) const {
  const Sophus::SE3d pose0(state0.rot, state0.pos);
  const Sophus::SE3d pose1(state1.rot, state1.pos);

  const Eigen::Vector3d pos =
      (1 - ratio) * pose0.translation() + ratio * pose1.translation();
  const Sophus::SO3d rot = Sophus::SO3d(
      pose0.unit_quaternion().slerp(ratio, pose1.unit_quaternion()));

  return Sophus::SE3d(rot, pos);
}

void LioEskf::save_map(const std::string &map_file) {
  ndt_lo_.save_map(map_file);
}
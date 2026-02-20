#include "eskf/ieskf.h"
#include <glog/logging.h>

bool IESKF::update_time(const double time) {
  if (timestamp_ > time) {
    LOG(INFO) << "time " << time << " is older than current: " << timestamp_;
    return false;
  }
  timestamp_ = time;
  nominal_state_.timestamp = time;
  error_state_.timestamp = time;
  return true;
}

void IESKF::initialize_noise(const ImuInitializer &initializer) {
  // noise_.dia_var_g_ = initializer.var_g();
  // noise_.dia_var_a_ = initializer.var_a();
  // ref
  noise_.dia_var_g_ =
      Eigen::Vector3d::Ones() * std::sqrt(initializer.var_g()[0]);
  noise_.dia_var_a_ =
      Eigen::Vector3d::Ones() * std::sqrt(initializer.var_a()[0]);
  nominal_state_.gravity = initializer.gravity();
  nominal_state_.vel = Eigen::Vector3d::Zero(); // static status
  nominal_state_.bias_a = initializer.bias_a();
  nominal_state_.bias_g = initializer.bias_g();
  cov_ = Eigen::Matrix<double, 18, 18>::Identity() * 1e-4;
  set_init_noise_ = true;
  update_time(initializer.timestamp());
  LOG(INFO) << "Noise initialized.";
}

void IESKF::initialize_pose(const Sophus::SE3d &Tob, const double timestamp) {
  nominal_state_.pos = Tob.translation();
  nominal_state_.rot = Tob.so3();
  set_init_pose_ = true;
  update_time(timestamp);
  LOG(INFO) << "Pose initialized.";
}

bool IESKF::predict_imu(const IMU &imu_data) {
  const double dt = imu_data.timestamp - nominal_state_.timestamp;
  if (timestamp_ == 0 || dt <= 0) {
    LOG(INFO) << "Invalid time interval. Skip current data.";
    update_time(imu_data.timestamp);
    return false;
  }
  IMUState predicted_state;
  predicted_state.pos =
      nominal_state_.pos + nominal_state_.vel * dt +
      0.5 * (nominal_state_.rot * (imu_data.acc - nominal_state_.bias_a)) * dt *
          dt +
      0.5 * nominal_state_.gravity * dt * dt;
  predicted_state.vel =
      nominal_state_.vel +
      nominal_state_.rot * (imu_data.acc - nominal_state_.bias_a) * dt +
      nominal_state_.gravity * dt;
  predicted_state.rot =
      nominal_state_.rot *
      Sophus::SO3d::exp((imu_data.gyr - nominal_state_.bias_g) * dt);
  // ba, bg, g unchanged

  Eigen::Matrix<double, 18, 18> F =
      Eigen::Matrix<double, 18, 18>::Identity(); // jacobian of motion model
  F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
  F.block<3, 3>(3, 6) =
      -(nominal_state_.rot.matrix() *
        Sophus::SO3d::hat(imu_data.acc - nominal_state_.bias_a) * dt);
  F.block<3, 3>(3, 12) = -nominal_state_.rot.matrix() * dt;
  F.block<3, 3>(3, 15) = Eigen::Matrix3d::Identity() * dt;
  F.block<3, 3>(6, 6) =
      Sophus::SO3d::exp(-(imu_data.gyr - nominal_state_.bias_g) * dt).matrix();
  F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;

  // error state unchanged
  // IMUState predict_err_state(F * error_state_.vec());

  // Eigen::Matrix<double, 18, 18> Q =
  //     Eigen::Matrix<double, 18, 18>::Zero(); // covariance of motion noise
  // Q.diagonal() << 0, 0, 0, noise_.dia_var_a_ * dt * dt,
  //     noise_.dia_var_g_ * dt * dt, noise_.var_bg_ * dt, noise_.var_bg_ * dt,
  //     noise_.var_bg_ * dt, noise_.var_ba_ * dt, noise_.var_ba_ * dt,
  //     noise_.var_ba_ * dt, 0, 0, 0;
  // ref
  Eigen::Matrix<double, 18, 18> Q =
      Eigen::Matrix<double, 18, 18>::Zero(); // covariance of motion noise
  Q.diagonal() << 0, 0, 0, noise_.dia_var_a_, noise_.dia_var_g_, noise_.var_bg_,
      noise_.var_bg_, noise_.var_bg_, noise_.var_ba_, noise_.var_ba_,
      noise_.var_ba_, 0, 0, 0;

  cov_ = F * cov_.eval() * F.transpose() + Q;
  nominal_state_.rot = predicted_state.rot;
  nominal_state_.pos = predicted_state.pos;
  nominal_state_.vel = predicted_state.vel;
  update_time(imu_data.timestamp);

  return true;
}

bool IESKF::correct_state() {
  nominal_state_.pos += error_state_.pos;
  nominal_state_.vel += error_state_.vel;
  nominal_state_.rot *= error_state_.rot;
  nominal_state_.bias_g += error_state_.bias_g;
  nominal_state_.bias_a += error_state_.bias_a;
  nominal_state_.gravity += error_state_.gravity;

  return true;
}

bool IESKF::correct_pose(IESKF::NDT_callback compute_Hb) {
  if (!compute_Hb) {
    LOG(WARNING) << "Callback function does not exist.";
    return false;
  }
  const Sophus::SO3d rot_0 = nominal_state_.rot;
  Mat18 P_0 = cov_;
  Mat18 P_k = Mat18::Zero();
  Mat18 PH_k = Mat18::Zero();
  // HtVinvH
  Mat18 H = Mat18::Zero();
  // HtVinv(z - h)
  Vec18 b = Vec18::Zero();

  for (size_t i = 0; i < params_.iterations; ++i) {
    const Eigen::Vector3d dtheta = (nominal_state_.rot.inverse() * rot_0).log();
    Mat18 J_k = Mat18::Identity();
    J_k.block<3, 3>(6, 6) =
        Eigen::Matrix3d::Identity() - 0.5 * Sophus::SO3d::hat(dtheta);
    P_k = J_k * P_0 * J_k.transpose();

    compute_Hb(state_SE3(), H, b);
    H = H * params_.scaling;
    b = b * params_.scaling;

    PH_k = (P_k.inverse() + H).inverse();
    Vec18 err_state = PH_k * b;
    error_state_ = IMUState(err_state);

    correct_state();

    LOG(INFO) << "It " << i << " err state norm: " << err_state.norm();

    if (err_state.norm() < params_.eps) {
      break;
    }
  }
  cov_ = (Mat18::Identity() - PH_k * H) * P_k;

  reset_error();

  return true;
}

bool IESKF::reset_error() {
  Eigen::Matrix<double, 18, 18> J =
      Eigen::Matrix<double, 18, 18>::Identity(); // jacobian of reset function
  J.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() -
                        0.5 * Sophus::SO3d::hat(error_state_.rot.log());
  cov_ = J * cov_ * J.transpose();
  error_state_ = IMUState(Eigen::Matrix<double, 18, 1>::Zero());
  error_state_.rot = Sophus::SO3d();
  return true;
}

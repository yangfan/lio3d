#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "tools/ImuInitializer.h"
#include "tools/imu.h"

#include <functional>

constexpr double kdeg2rad_ = M_PI / 180.0;

class IESKF {
public:
  using Mat18 = Eigen::Matrix<double, 18, 18>;
  using Vec18 = Eigen::Matrix<double, 18, 1>;
  using NDT_callback =
      std::function<void(const Sophus::SE3d &T_W_I, Mat18 &H, Vec18 &b)>;
  struct Noise {
    double var_bg_ = 1e-6; // contineous: rad / (s * sqrt(s))
    double var_ba_ = 1e-4; // continuous: m / (s^2 * sqr(s))
    Eigen::Vector3d dia_var_g_ = Eigen::Vector3d::Zero(); // discrete rad / s
    Eigen::Vector3d dia_var_a_ = Eigen::Vector3d::Zero(); // discrete m / (s*s)
  };
  struct Params {
    size_t iterations = 5;
    double eps = 1e-3;
    double scaling = 0.01;
  };

  bool noise_initialized() const { return set_init_noise_; }
  bool pose_initialized() const { return set_init_pose_; }

  IMUState state() const { return nominal_state_; }
  Sophus::SE3d state_SE3() const {
    return Sophus::SE3d(nominal_state_.rot, nominal_state_.pos);
  }

  void initialize_noise(const ImuInitializer &initializer);
  void initialize_pose(const Sophus::SE3d &Tob, const double timestamp);

  bool predict_imu(const IMU &imu_data);
  bool correct_pose(NDT_callback compute_Hb);

  bool correct_state();
  bool reset_error();

  bool update_time(const double time);

  void set_state(const IMUState state) { nominal_state_ = state; }

private:
  IMUState nominal_state_;
  IMUState error_state_;
  Eigen::Matrix<double, 18, 18> cov_ = Eigen::Matrix<double, 18, 18>::Zero();

  Noise noise_;
  double timestamp_ = 0.0;

  bool set_init_pose_ = false;
  bool set_init_noise_ = false;

  Params params_;
};
#include "KDTree.hpp"
#include "NDT.h"

#include <algorithm>
#include <chrono>
#include <execution>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

DEFINE_string(data_path, "/home/fan/ssd/Projects/ros2_ws/src/lio3d/data/",
              "point cloud file path");

using pcl_Point = pcl::PointXYZI;
using pcl_PointCloud = pcl::PointCloud<pcl_Point>;

template <typename C, int dim, typename Getter>
void ComputeMeanAndCov(const C &data, Eigen::Matrix<double, dim, 1> &mean,
                       Eigen::Matrix<double, dim, dim> &cov, Getter &&getter) {
  using D = Eigen::Matrix<double, dim, 1>;
  using E = Eigen::Matrix<double, dim, dim>;
  size_t len = data.size();
  assert(len > 1);

  mean = std::accumulate(data.begin(), data.end(),
                         Eigen::Matrix<double, dim, 1>::Zero().eval(),
                         [&getter](const D &sum, const auto &data) -> D {
                           return sum + getter(data);
                         }) /
         len;
  cov = std::accumulate(data.begin(), data.end(), E::Zero().eval(),
                        [&mean, &getter](const E &sum, const auto &data) -> E {
                          auto value = getter(data).eval();
                          D v = value - mean;
                          return sum + v * v.transpose();
                        }) /
        (len - 1);
}

pcl_PointCloud::Ptr DownSampling(pcl_PointCloud::Ptr origin,
                                 const float voxel_sz) {
  pcl::VoxelGrid<pcl_Point> grid;
  grid.setLeafSize(voxel_sz, voxel_sz, voxel_sz);
  grid.setInputCloud(origin);

  pcl_PointCloud::Ptr output(new pcl_PointCloud);
  grid.filter(*output);
  return output;
}

TEST(NDT, Build) {
  NDT::PointCloudPtr target(new pcl_PointCloud);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *target);
  LOG(INFO) << "Number of points: " << target->size();

  NDT ndt;

  auto start = std::chrono::steady_clock::now();
  ndt.set_target_cloud(target);
  auto end = std::chrono::steady_clock::now();
  const double elapse =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "Took " << elapse << " ms to build voxel grids.";

  const auto grid = ndt.grid();
  LOG(INFO) << "grid size: " << grid.size();

  std::for_each(
      std::execution::par_unseq, grid.cbegin(), grid.cend(),
      [&](const std::pair<NDT::VoxelId, NDT::Voxel> &v) {
        if (v.second.pids.size() > 5) {

          Eigen::Vector3d mean;
          Eigen::Matrix3d cov;
          Eigen::Matrix3d info;
          ComputeMeanAndCov(
              v.second.pids, mean, cov, [&target](const size_t &idx) {
                return (target->points[idx].getVector3fMap().cast<double>());
              });

          Eigen::JacobiSVD svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
          Eigen::Vector3d lambda = svd.singularValues();
          if (lambda[1] < lambda[0] * 1e-3) {
            lambda[1] = lambda[0] * 1e-3;
          }

          if (lambda[2] < lambda[0] * 1e-3) {
            lambda[2] = lambda[0] * 1e-3;
          }

          Eigen::Matrix3d inv_lambda =
              Eigen::Vector3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2])
                  .asDiagonal();
          info = svd.matrixV() * inv_lambda * svd.matrixU().transpose();

          EXPECT_LT((mean - v.second.mean).norm(), 1e-2);
          EXPECT_LT((cov - v.second.cov).norm(), 1e-2);
          EXPECT_LT((info - v.second.info).norm(), 1e-2);
        }
      });
}

TEST(NDT, KNN) {
  pcl_PointCloud::Ptr cloud1(new pcl_PointCloud);
  //   pcl::io::loadPCDFile(FLAGS_data_path + "input/first.pcd", *cloud1);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *cloud1);
  LOG(INFO) << "Number of points1: " << cloud1->size();

  pcl_PointCloud::Ptr cloud2(new pcl_PointCloud);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *cloud2);
  LOG(INFO) << "Number of points2: " << cloud2->size();

  cloud1 = DownSampling(cloud1, 0.01);
  cloud2 = DownSampling(cloud2, 0.01);
  LOG(INFO) << "Downsampled point cloud1 size: " << cloud1->size();
  LOG(INFO) << "Downsampled point cloud2 size: " << cloud2->size();

  NDT::PointCloudPtr target = cloud1;
  NDT::PointCloudPtr source = cloud2;
  LOG(INFO) << "number of query points: " << source->size();

  pcl::search::KdTree<pcl_Point> kdtree;
  kdtree.setInputCloud(cloud1);

  std::vector<std::vector<int>> pcl_idx;
  std::vector<std::vector<float>> pcl_dist;
  std::vector<int> qidx(cloud2->size());
  std::iota(qidx.begin(), qidx.end(), 0);

  auto start = std::chrono::steady_clock::now();
  kdtree.nearestKSearch(*cloud2, qidx, 5, pcl_idx, pcl_dist);
  auto end = std::chrono::steady_clock::now();
  double elapse =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "pcl kd k cloud search took " << elapse << " ms.";

  KDTree<double, 3>::PointCloud kd_target;
  kd_target.reserve(cloud1->size());
  for (const auto &pt : cloud1->points) {
    kd_target.emplace_back(pt.getVector3fMap().cast<double>());
  }
  KDTree<double, 3>::PointCloud kd_source;
  kd_source.reserve(cloud2->size());
  for (size_t i = 0; i < cloud2->size(); ++i) {
    kd_source.emplace_back(cloud2->points[i].getVector3fMap().cast<double>());
  }

  KDTree<double, 3> kd;
  kd.setInputCloud(kd_target);
  std::vector<std::vector<int>> kd_idx;
  std::vector<std::vector<double>> kd_dist;
  start = std::chrono::steady_clock::now();
  kd.nearest_neighbors_kmt(kd_source, 5, kd_idx, kd_dist);
  end = std::chrono::steady_clock::now();
  elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count();
  LOG(INFO) << "kd k cloud search took " << elapse << " ms.";

  NDT ndt;
  NDT::Params params;
  params.nb_type = NDT::NeighborType::NB6;
  params.vx_size = 0.5;
  ndt.set_params(params);
  ndt.set_target_cloud(target);

  elapse = 0.0;
  for (size_t i = 0; i < cloud2->size(); ++i) {
    std::vector<int> nearest_idx;
    std::vector<double> nearest_dist;

    start = std::chrono::steady_clock::now();
    ndt.nearest_neighbors(source->points[i], 3, nearest_idx, nearest_dist);
    end = std::chrono::steady_clock::now();
    elapse += std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
    for (size_t j = 0; j < nearest_idx.size(); ++j) {
      EXPECT_LE(nearest_dist[j] - 1e-3, pcl_dist[i][j]);
    }
  }
  LOG(INFO) << "voxel k point search took average: " << elapse / cloud2->size()
            << " ms.";

  std::vector<std::vector<int>> ndt_idx;
  std::vector<std::vector<double>> ndt_dist;

  start = std::chrono::steady_clock::now();
  ndt.nearest_neighbors_kmt(source, 5, ndt_idx, ndt_dist);
  end = std::chrono::steady_clock::now();
  elapse = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count();
  LOG(INFO) << "ndt k cloud search took " << elapse << " ms.";

  for (size_t i = 0; i < ndt_idx.size(); ++i) {
    for (size_t j = 0; j < ndt_idx[i].size(); ++j) {
      EXPECT_LE(ndt_dist[i][j] - 1e-3, pcl_dist[i][j]);
      EXPECT_LE(ndt_dist[i][j] - 1e-3, kd_dist[i][j]);
    }
  }
}

TEST(NDT, Align) {

  pcl::PointCloud<pcl::PointXYZI>::Ptr tc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr sc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *tc);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/source.pcd", *sc);

  NDT::PointCloudPtr target = tc;
  NDT::PointCloudPtr source = sc;
  LOG(INFO) << "size of source: " << source->size();
  LOG(INFO) << "size of target: " << target->size();

  NDT ndt;
  NDT::Params params;
  params.nb_type = NDT::NeighborType::NB0;
  params.vx_size = 0.5;
  ndt.set_params(params);
  ndt.set_target_cloud(target);
  ndt.set_source_cloud(source);

  Sophus::SE3d Tts;

  auto start = std::chrono::steady_clock::now();
  EXPECT_TRUE(ndt.align(Tts));
  auto end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "NDT alignment took " << elapsed << " ms."; // 38 ms

  std::ifstream ifs(FLAGS_data_path + "/input/ground_truth.txt");
  Sophus::SE3d ground_truth;
  if (ifs.is_open()) {
    double tx, ty, tz, qw, qx, qy, qz;
    ifs >> tx >> ty >> tz >> qw >> qx >> qy >> qz;
    ground_truth = Sophus::SE3d(Eigen::Quaterniond(qw, qx, qy, qz),
                                Eigen::Vector3d(tx, ty, tz));
  }
  const double pose_err = (ground_truth.inverse() * Tts).log().norm();
  LOG(INFO) << "pose err: " << pose_err;
  EXPECT_LT(pose_err, 1e-2);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_aligned(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*sc, *cloud_aligned, Tts.matrix().cast<float>());
  cloud_aligned->height = 1;
  cloud_aligned->width = cloud_aligned->size();
  pcl::io::savePCDFile(FLAGS_data_path + "output/ndt_aligned.pcd",
                       *cloud_aligned);

  tc->height = 1;
  tc->width = tc->size();
  pcl::io::savePCDFile(FLAGS_data_path + "output/target.pcd", *tc);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_gt(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*sc, *cloud_gt, ground_truth.matrix().cast<float>());
  cloud_gt->height = 1;
  cloud_gt->width = cloud_aligned->size();
  pcl::io::savePCDFile(FLAGS_data_path + "output/gt.pcd", *cloud_gt);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}
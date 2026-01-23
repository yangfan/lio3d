#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <sophus/se3.hpp>

#include <chrono>
#include <fstream>

#include "ICP3D.h"

DEFINE_string(data_path, "/home/fan/ssd/Projects/ros2_ws/src/lio3d/data/",
              "point cloud file path");

TEST(ICP, Point2Point) {

  pcl::PointCloud<pcl::PointXYZI>::Ptr tc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr sc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *tc);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/source.pcd", *sc);

  PointCloud3D target, source;
  target.reserve(tc->size());
  source.reserve(sc->size());
  std::for_each(tc->points.begin(), tc->points.end(),
                [&target](const pcl::PointXYZI &pt) {
                  target.emplace_back(pt.getVector3fMap().cast<double>());
                });
  std::for_each(sc->points.begin(), sc->points.end(),
                [&source](const pcl::PointXYZI &ps) {
                  source.emplace_back(ps.getVector3fMap().cast<double>());
                });
  EXPECT_EQ(target.size(), tc->size());
  EXPECT_EQ(source.size(), sc->size());

  ICP3D icp(ICP3D::Params(20, 1.0, 10));
  icp.set_target_cloud(std::move(target));
  icp.set_source_cloud(std::move(source));

  Sophus::SE3d Tts;

  auto start = std::chrono::steady_clock::now();
  EXPECT_TRUE(icp.align_p2p(Tts));
  auto end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "ICP point to point took " << elapsed << " ms."; // 233 ms

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
  pcl::io::savePCDFile(FLAGS_data_path + "output/aligned_p2p.pcd",
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

TEST(ICP, Point2Line) {

  pcl::PointCloud<pcl::PointXYZI>::Ptr tc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr sc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *tc);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/source.pcd", *sc);

  PointCloud3D target, source;
  target.reserve(tc->size());
  source.reserve(sc->size());
  std::for_each(tc->points.begin(), tc->points.end(),
                [&target](const pcl::PointXYZI &pt) {
                  target.emplace_back(pt.getVector3fMap().cast<double>());
                });
  std::for_each(sc->points.begin(), sc->points.end(),
                [&source](const pcl::PointXYZI &ps) {
                  source.emplace_back(ps.getVector3fMap().cast<double>());
                });
  EXPECT_EQ(target.size(), tc->size());
  EXPECT_EQ(source.size(), sc->size());
  LOG(INFO) << "number target point cloud: " << target.size();
  LOG(INFO) << "number source point cloud: " << source.size();

  ICP3D icp(ICP3D::Params(20, 1.0, 10));
  icp.set_target_cloud(std::move(target));
  icp.set_source_cloud(std::move(source));

  Sophus::SE3d Tts;

  auto start = std::chrono::steady_clock::now();
  EXPECT_TRUE(icp.align_p2l(Tts));
  auto end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "ICP point to line took " << elapsed << " ms."; // 369 ms

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
  pcl::io::savePCDFile(FLAGS_data_path + "output/aligned_p2l.pcd",
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

TEST(ICP, Point2Plane) {

  pcl::PointCloud<pcl::PointXYZI>::Ptr tc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr sc(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/target.pcd", *tc);
  pcl::io::loadPCDFile(FLAGS_data_path + "input/source.pcd", *sc);

  PointCloud3D target, source;
  target.reserve(tc->size());
  source.reserve(sc->size());
  std::for_each(tc->points.begin(), tc->points.end(),
                [&target](const pcl::PointXYZI &pt) {
                  target.emplace_back(pt.getVector3fMap().cast<double>());
                });
  std::for_each(sc->points.begin(), sc->points.end(),
                [&source](const pcl::PointXYZI &ps) {
                  source.emplace_back(ps.getVector3fMap().cast<double>());
                });
  EXPECT_EQ(target.size(), tc->size());
  EXPECT_EQ(source.size(), sc->size());
  LOG(INFO) << "number target point cloud: " << target.size();
  LOG(INFO) << "number source point cloud: " << source.size();

  ICP3D icp(ICP3D::Params(20, 1.0, 10));
  icp.set_target_cloud(std::move(target));
  icp.set_source_cloud(std::move(source));

  Sophus::SE3d Tts;

  auto start = std::chrono::steady_clock::now();
  EXPECT_TRUE(icp.align_p2pl(Tts));
  auto end = std::chrono::steady_clock::now();
  double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  LOG(INFO) << "ICP point to plane took " << elapsed << " ms."; // 97 ms

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
  pcl::io::savePCDFile(FLAGS_data_path + "output/aligned_p2pl.pcd",
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
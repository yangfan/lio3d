#include "tools/BagIO.h"
#include "tools/LidarPointType.h"
#include "tools/PointCloudConverter.h"
#include "tools/Sync.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

DEFINE_string(bag_file,
              // "/home/fan/ssd/Projects/ros2_ws/data/bags/ULHK/ros2_cloud2",
              "/home/fan/ssd/Projects/ros2_ws/data/bags/NCLT/ros2_0110",
              "path to ros2bag file");
DEFINE_string(
    config_file,
    // "/home/fan/ssd/Projects/ros2_ws/src/lio3d/config/velodyne_ulhk.yaml",
    "/home/fan/ssd/Projects/ros2_ws/src/lio3d/config/velodyne_nclt.yaml",
    "path of configuration file");
// DEFINE_string(topic_name, "",
DEFINE_string(cloud_topic_name,
              // "/velodyne_points_0",
              "points_raw", "topic name of pointcloud data");

DEFINE_string(imu_topic_name,
              // "/imu/data",
              "imu_raw", "topic name of imu data");

DEFINE_string(data_path,
              "/home/fan/ssd/Projects/ros2_ws/src/lio3d/data/output/",
              "Path to save map file");

// TEST(Data, Convert) {
//   BagIO bag_io(FLAGS_bag_file);
//   PointCloudConverter convert;
//   convert.config(FLAGS_config_file);
//   bag_io
//       .AddPointCloudHandle(
//           FLAGS_cloud_topic_name,
//           [&](std::unique_ptr<sensor_msgs::msg::PointCloud2> cloud) {
//             LidarPointCloudPtr pcl_cloud(new LidarPointCloud);
//             LOG(INFO) << "start time: " << cloud->header.stamp.sec;
//             convert.convert(*cloud, pcl_cloud);
//             LOG(INFO) << "number of pt: " << pcl_cloud->points.size();
//             LOG(INFO) << std::setprecision(15) << "start time offset: "
//                       << pcl_cloud->points.front().time;
//             LOG(INFO) << std::setprecision(15)
//                       << "end time offset: " <<
//                       pcl_cloud->points.back().time;
//             return true;
//           })
//       .AddIMUHandle(FLAGS_imu_topic_name,
//                     [&](std::unique_ptr<sensor_msgs::msg::Imu> imu) {
//                       LOG(INFO) << "gyro: " << imu->angular_velocity.x << ",
//                       "
//                                 << imu->angular_velocity.y << ", "
//                                 << imu->angular_velocity.z;
//                       LOG(INFO) << "acc: " << imu->linear_acceleration.x <<
//                       ", "
//                                 << imu->linear_acceleration.y << ", "
//                                 << imu->linear_acceleration.z;
//                       return true;
//                     })
//       .Process();
// }

TEST(Data, Sync) {
  BagIO bag_io(FLAGS_bag_file);
  Sync sync;
  sync.config(FLAGS_config_file);
  bag_io
      .AddPointCloudHandle(
          FLAGS_cloud_topic_name,
          [&](std::unique_ptr<sensor_msgs::msg::PointCloud2> cloud_msg) {
            sync.add_cloud(std::move(cloud_msg));
            return true;
          })
      .AddIMUHandle(FLAGS_imu_topic_name,
                    [&](std::unique_ptr<sensor_msgs::msg::Imu> imu_msg) {
                      sync.add_imu(std::move(imu_msg));
                      return true;
                    })
      .Process();
}
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}
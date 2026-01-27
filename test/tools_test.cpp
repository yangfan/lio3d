#include "tools/BagIO.h"
#include "tools/MapViewer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/pointcloud/ros2_cloud2",
              "path to ros2bag file");
DEFINE_string(topic_name, "/velodyne_points_0",
              "topic name of pointcloud data");

TEST(Tools, Viewer) {
  MapViewer viewer("Map Viewer", 0.5f);
  BagIO bag_io(FLAGS_bag_file);
  bag_io
      .AddPointCloudHandle(
          FLAGS_topic_name,
          [&viewer](std::unique_ptr<sensor_msgs::msg::PointCloud2> cloud) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(
                new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*cloud, *pcl_cloud);
            viewer.add_pointcloud(pcl_cloud, Sophus::SE3d());
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
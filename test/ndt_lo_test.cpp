#include "lio/ndt_lo.h"
#include "tools/BagIO.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <chrono>

DEFINE_string(bag_file,
              "/home/fan/ssd/Projects/ros2_ws/data/bags/ULHK/ros2_cloud2",
              "path to ros2bag file");
DEFINE_string(topic_name, "/velodyne_points_0",
              "topic name of pointcloud data");
DEFINE_string(data_path,
              "/home/fan/ssd/Projects/ros2_ws/src/lio3d/data/output/",
              "Path to save map file");
DEFINE_bool(visualizer_on, true, "turn on visualizer");

NDT_LO::PointCloud::Ptr DownSampling(NDT_LO::PointCloud::Ptr origin,
                                     const float voxel_sz) {
  pcl::VoxelGrid<NDT_LO::Point> grid;
  grid.setLeafSize(voxel_sz, voxel_sz, voxel_sz);
  grid.setInputCloud(origin);

  NDT_LO::PointCloud::Ptr output(new NDT_LO::PointCloud);
  grid.filter(*output);
  return output;
}

TEST(NDT_LO, BagTest) {
  NDT_LO::Params params;
  params.viwer_on = FLAGS_visualizer_on;
  params.ndt_params.nb_type = NDT::NeighborType::NB0;
  params.ndt_params.vx_size = 1;
  params.ndt_params.min_vx_pt = 4;
  params.ndt_params.guess_translation = false;

  NDT_LO ndt_lo(params);

  double elapsed = 0.0;
  size_t cnt = 0;
  BagIO bag_io(FLAGS_bag_file);
  bag_io
      .AddPointCloudHandle(
          FLAGS_topic_name,
          [&ndt_lo, &elapsed,
           &cnt](std::unique_ptr<sensor_msgs::msg::PointCloud2> cloud) {
            NDT_LO::PointCloud::Ptr scan(new NDT_LO::PointCloud);
            pcl::fromROSMsg(*cloud, *scan);
            auto start = std::chrono::steady_clock::now();
            ndt_lo.add_scan(DownSampling(scan, 0.1));
            auto end = std::chrono::steady_clock::now();
            elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - start)
                           .count();
            cnt++;

            return true;
          })
      .Process();
  // 165879 ms total
  // average 26.7653 ms without visualizer
  // average 49.1621 ms with visualizer
  LOG(INFO) << "Frame processing took average " << elapsed / cnt << " ms.";
  ndt_lo.save_map(FLAGS_data_path + "ndt_lo_map.pcd");
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  google::ParseCommandLineFlags(&argc, &argv, true);

  return RUN_ALL_TESTS();
}
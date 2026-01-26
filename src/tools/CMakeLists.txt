add_library(tools MapViewer.cpp BagIO.cpp)
target_link_libraries(tools ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus rclcpp::rclcpp rosbag2_transport::rosbag2_transport ${sensor_msgs_TARGETS})
target_include_directories(tools PUBLIC ${CMAKE_SOURCE_DIR}/include/lio3d/tools)
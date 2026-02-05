add_library(eskf LioEskf.cpp eskf.cpp)
target_link_libraries(eskf tools lo yaml-cpp::yaml-cpp Eigen3::Eigen Sophus::Sophus ${PCL_LIBRARIES} rclcpp::rclcpp ${sensor_msgs_TARGETS})
target_include_directories(eskf PRIVATE ${CMAKE_SOURCE_DIR}/include/lio3d)

add_library(ieskf LioIeskf.cpp ieskf.cpp)
target_link_libraries(ieskf tools NDT_INC yaml-cpp::yaml-cpp Eigen3::Eigen Sophus::Sophus ${PCL_LIBRARIES} rclcpp::rclcpp ${sensor_msgs_TARGETS})
target_include_directories(ieskf PRIVATE ${CMAKE_SOURCE_DIR}/include/lio3d)
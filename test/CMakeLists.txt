enable_testing()

add_executable(icp_test icp_test.cpp)
target_link_libraries(icp_test ICP gtest glog::glog ${GFLAGS_LIBRARIES} Sophus::Sophus ${PCL_LIBRARIES})
target_include_directories(icp_test PRIVATE ${CMAKE_SOURCE_DIR}/include/lio3d/matching)

add_executable(ndt_test ndt_test.cpp)
target_link_libraries(ndt_test NDT gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus tbb)
target_include_directories(ndt_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include/lio3d/matching
)

add_executable(tools_test tools_test.cpp)
target_link_libraries(tools_test NDT tools pcl_conversions::pcl_conversions gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus tbb)
target_include_directories(tools_test PRIVATE ${CMAKE_SOURCE_DIR}/include/lio3d)

add_executable(ndt_lo_test ndt_lo_test.cpp)
target_link_libraries(ndt_lo_test lo NDT tools pcl_conversions::pcl_conversions gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus)
target_include_directories(ndt_lo_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include/lio3d
)

add_executable(ndt_inc_lo_test ndt_inc_lo_test.cpp)
target_link_libraries(ndt_inc_lo_test lo NDT tools pcl_conversions::pcl_conversions gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus)
target_include_directories(ndt_inc_lo_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include/lio3d
)

add_executable(data_process data_process_test.cpp)
target_link_libraries(data_process tools pcl_conversions::pcl_conversions gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus)
target_include_directories(data_process PRIVATE
    ${CMAKE_SOURCE_DIR}/include/lio3d
)

add_executable(eskf_test eskf_test.cpp)
target_link_libraries(eskf_test eskf tools pcl_conversions::pcl_conversions gtest glog::glog ${GFLAGS_LIBRARIES} ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus)
target_include_directories(eskf_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include/lio3d
)

include(GoogleTest)
gtest_discover_tests(icp_test ndt_test tools_test)
enable_testing()

add_executable(icp_test icp_test.cpp)
target_link_libraries(icp_test ICP gtest glog::glog ${GFLAGS_LIBRARIES} Sophus::Sophus ${PCL_LIBRARIES})
target_include_directories(icp_test PRIVATE ${CMAKE_SOURCE_DIR}/include/lio3d/matching)

include(GoogleTest)
gtest_discover_tests(icp_test)
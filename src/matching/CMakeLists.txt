add_library(ICP ICP3D.cpp)
target_link_libraries(ICP PUBLIC Sophus::Sophus Eigen3::Eigen TBB::tbb)
target_include_directories(ICP PUBLIC ${CMAKE_SOURCE_DIR}/include/lio3d/matching)

add_library(NDT NDT.cpp)
target_link_libraries(NDT PUBLIC Eigen3::Eigen Sophus::Sophus TBB::tbb)
target_include_directories(NDT PUBLIC ${CMAKE_SOURCE_DIR}/include/lio3d/matching)
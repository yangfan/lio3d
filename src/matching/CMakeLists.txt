add_library(ICP ICP3D.cpp)

target_link_libraries(ICP PUBLIC Sophus::Sophus Eigen3::Eigen tbb)

target_include_directories(ICP PUBLIC ${CMAKE_SOURCE_DIR}/include/lio3d/matching)
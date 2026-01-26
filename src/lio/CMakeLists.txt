add_library(lo ndt_lo.cpp)
target_link_libraries(lo NDT tools ${PCL_LIBRARIES} Eigen3::Eigen Sophus::Sophus)
target_include_directories(lo PUBLIC ${CMAKE_SOURCE_DIR}/include/lio3d/)
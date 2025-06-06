cmake_minimum_required(VERSION 3.10)
project(TarTanAirAugmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_PREFIX_PATH "/opt/ros/noetic" ${CMAKE_PREFIX_PATH})

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cnpy REQUIRED)  # 通过 find_package 找 cnpy
find_package(PCL REQUIRED COMPONENTS common io)

include_directories(${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  /opt/ros/noetic/include/cnpy
  ${PCL_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ./src/augmentation.cpp)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  /opt/ros/noetic/lib/libcnpy.so
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)


cmake_minimum_required(VERSION 3.15)

project(radial_sweep)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(WITH_QT OFF)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_compile_options(-Wall -Wextra -pedantic)
add_executable(radial_sweep mq06861.cpp)
target_link_libraries(radial_sweep ${OpenCV_LIBS})
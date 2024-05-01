cmake_minimum_required(VERSION 3.15)

project(demosaic)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(WITH_QT OFF)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_compile_options(-Wall -Wextra -pedantic)
add_executable(demosaic mq06861.cpp)
target_link_libraries(demosaic ${OpenCV_LIBS})
cmake_minimum_required(VERSION 3.6)
project(interpolation_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")

set(SOURCE_FILES coeff.c interpol.c io.c demo.c)

add_executable(interpolation_demo ${SOURCE_FILES})
target_link_libraries(interpolation_demo m)

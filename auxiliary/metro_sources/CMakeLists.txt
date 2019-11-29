cmake_minimum_required(VERSION 3.0)
project(metro)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")

add_executable(metro metro.cpp plylib.cpp)

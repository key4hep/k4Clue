cmake_minimum_required(VERSION 3.14...3.18)

# Location of the CUDA Toolkit
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "g++")

# Location of the cupla
set(cupla_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cupla)

project(
  CLUE
  VERSION 1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CUDA CXX)

include(GNUInstallDirs)

# Set up C++ Standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if(NOT CMAKE_CXX_STANDARD MATCHES "14|17|20")
  message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

#Disable tests when ready
#include(CTest)

find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(src)

message(STATUS "Finished!")


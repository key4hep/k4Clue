cmake_minimum_required(VERSION 3.14...3.18)

# Location of the cupla
set(cupla_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cupla)

# Location of TBB
set(TBB_BASE "/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/tbb/2019_U8")

project(
  CLUE
  VERSION 1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CXX)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Set up CUDA")

  # Location of the CUDA Toolkit
  set(CMAKE_CUDA_HOST_COMPILER "g++")
  set(CMAKE_CUDA_ARCHITECTURES "60;70;75")

  enable_language(CUDA)
else()
  message(STATUS "CUDA not found. Still, you can run the C++ version!")
endif()

include(GNUInstallDirs)

# Set up C++ Standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if(NOT CMAKE_CXX_STANDARD MATCHES "14|17|20")
  message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost package found!")
endif()

find_package(TBB REQUIRED)
if(TBB_FOUND)
  message(STATUS "TBB package found!")
endif()

add_subdirectory(src)

#Disable tests when ready
#include(CTest)

message(STATUS "Finished!")


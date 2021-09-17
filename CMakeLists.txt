cmake_minimum_required(VERSION 3.8)

# Location of the cupla
set(cupla_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cupla)

project(
  CLUE
  VERSION 1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CXX)

# Location of the CUDA Toolkit
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Set up CUDA")

  set(CMAKE_CUDA_HOST_COMPILER "g++")
  set(CMAKE_CUDA_ARCHITECTURES "60;70;75")

  enable_language(CUDA)
else()
  message(STATUS "CUDA not found. Still, you can run the C++ version!")
endif()

include(GNUInstallDirs)

# Set up C++ Standard
set(CMAKE_CXX_STANDARD 14 CACHE STRING "")

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

find_package(EDM4HEP)
if(EDM4HEP_FOUND)
  message(STATUS "EDM4HEP package found!")
endif()
find_package(DD4hep)
if(DD4HEP_FOUND)
  message(STATUS "DD4HEP package found!")
endif()
find_package(k4FWCore REQUIRED)
find_package(Gaudi REQUIRED)
gaudi_install(PYTHON)
gaudi_install(SCRIPTS)

add_subdirectory(src)

#Disable tests when ready
#include(CTest)

message(STATUS "Finished!")


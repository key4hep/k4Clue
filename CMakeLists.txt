cmake_minimum_required(VERSION 3.14...3.18)

project(
  CLUE
  VERSION 1.0
  DESCRIPTION "The CLUstering by Energy algorithm"
  LANGUAGES CXX)

include(GNUInstallDirs)

# Set up C++ Standard
# ``-DCMAKE_CXX_STANDARD=<standard>`` when invoking CMake
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


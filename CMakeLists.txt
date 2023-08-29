#[[
Copyright (c) 2020-2023 Key4hep-Project.

This file is part of Key4hep.
See https://key4hep.github.io/key4hep-doc/ for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]
cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW) # use package_ROOT env var to find packages

project(
  k4CLUE
  VERSION 1.0
  DESCRIPTION "k4CLUE: CLUE for future colliders calorimeters"
  LANGUAGES CXX)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE PATH "default install path" FORCE )
      endif()

include(GNUInstallDirs)

# Set up C++ Standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

if(NOT CMAKE_CXX_STANDARD MATCHES "17|20")
  message(FATAL_ERROR "Unsupported C++ standard: ${CMAKE_CXX_STANDARD}")
endif()

find_package(Boost REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost package found!")
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
if(k4FWCore_FOUND)
  message(STATUS "k4FWCore package found!")
endif()
find_package(Gaudi REQUIRED)
if(Gaudi_FOUND)
  message(STATUS "Gaudi package found!")
endif()

#gaudi_install(PYTHON)
#gaudi_install(SCRIPTS)

include(CTest)

add_subdirectory(src)

message(STATUS "Finished!")


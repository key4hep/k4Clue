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

# Optionally glob, but only for CMake 3.12 or later: file(GLOB HEADER_LIST
# CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/modern/*.hpp")
set(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")

set(CMAKE_CXX_FLAGS "-fPIC")

## Make an automatic library - will be static or dynamic based on user setting
add_library(CLUEAlgo_lib CLUEAlgo.cc ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(CLUEAlgo_lib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS CLUEAlgo_lib ${INSTALL_LIBRARIES}
  EXPORT CLUEAlgoTarget
  DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})

# CLUE as Gaudi algorithm
add_subdirectory(k4clue)

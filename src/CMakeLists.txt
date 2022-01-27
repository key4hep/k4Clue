# Optionally glob, but only for CMake 3.12 or later: file(GLOB HEADER_LIST
# CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/modern/*.hpp")
set(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")

################################################################################
set(CMAKE_CXX_FLAGS "-fPIC")

if(CMAKE_CUDA_COMPILER)
  ## internal flags for CUDA
  set(TARGET_SIZE "64")
  set(LDFLAGS " ")
  string(APPEND CMAKE_CUDA_FLAGS " -m${TARGET_SIZE}")
  string(APPEND CMAKE_CUDA_FLAGS " ${LDFLAGS}")
  
  # c++ flags
  set(CCFLAGS "-std=c++17")
  string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler ${CCFLAGS}")
  
  # other flags
  string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr -w")
endif(CMAKE_CUDA_COMPILER)

################################################################################

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
################################################################################


# Native C++ CLUE and native CUDA CLUE
add_subdirectory(clue)

# Native C++ CLUE and TBB CLUE using CUPLA
add_subdirectory(clue_tbb_cupla)

# Native C++ CLUE and CUDA CLUE using CUPLA
add_subdirectory(clue_cuda_cupla)

# CLUE as Gaudi algorithm
add_subdirectory(k4clue)

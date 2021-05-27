# Optionally glob, but only for CMake 3.12 or later: file(GLOB HEADER_LIST
# CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/modern/*.hpp")
set(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")

################################################################################

# internal flags for CUDA
set(TARGET_SIZE "64")
set(LDFLAGS " ")
string(APPEND CMAKE_CUDA_FLAGS " -m${TARGET_SIZE}")
string(APPEND CMAKE_CUDA_FLAGS " ${LDFLAGS}")

# c++ flags
set(CCFLAGS "-std=c++14")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler ${CCFLAGS}")

# other flags
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr -w")

# Gencode arguments
set(SMS 60;70;75)
# Generate SASS code for each SM architecture listed in $(SMS)
foreach(sms IN LISTS SMS)
  string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_${sms},code=compute_${sms}")
endforeach()

################################################################################

# Make an automatic library - will be static or dynamic based on user setting
add_library(CLUEAlgo_lib CLUEAlgo.cc ${HEADER_LIST})
add_library(CLUEAlgoGPU_lib CLUEAlgoGPU.cu ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(CLUEAlgo_lib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(CLUEAlgoGPU_lib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})

install(TARGETS CLUEAlgo_lib CLUEAlgoGPU_lib ${INSTALL_LIBRARIES}
  EXPORT CLUEAlgoTarget
  DESTINATION "${CMAKE_INSTALL_LIBDIR}")

################################################################################

# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.

# A copy with extension cu is needed to set the same source file in 
# both languages (C++/CUDA)
configure_file(main.cc main.cu)

set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
add_executable(main main.cu)

target_link_libraries(main PRIVATE CLUEAlgo_lib CLUEAlgoGPU_lib)

# mainCuplaCPUTBB: this will build the native C++ implementation of CLUE and
#       its corresponding TBB one built using CUPLA. Which one to use must be
#       selected at runtime via a flag (misleadingly enough called useGPU in
#       the help text).
set(tbb_lib -L${TBB_BASE}/lib -ltbb)

set_source_files_properties(main.cc PROPERTIES LANGUAGE CXX)
add_executable(mainCuplaCPUTBB main.cc)

target_compile_options(mainCuplaCPUTBB PRIVATE -DUSE_CUPLA -DFOR_TBB ${tbb_lib})
target_link_libraries(mainCuplaCPUTBB PRIVATE CLUEAlgo_lib tbb)

target_include_directories(mainCuplaCPUTBB PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/alpaka/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${TBB_BASE}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# mainCuplaCUDA: this will build the native C++ implementation of CLUE and its
#       corresponding CUDA one built using CUPLA. Which one to use must be
#       selected at runtime via a flag.
add_executable(mainCuplaCUDA main.cu)
target_compile_options(mainCuplaCUDA PRIVATE -DUSE_CUPLA -DFOR_CUDA -DALPAKA_ACC_GPU_CUDA_ENABLED=1)

target_link_libraries(mainCuplaCUDA PRIVATE CLUEAlgo_lib)

target_include_directories(mainCuplaCUDA PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/alpaka/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


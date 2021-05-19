# Optionally glob, but only for CMake 3.12 or later: file(GLOB HEADER_LIST
# CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/modern/*.hpp")
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/CLUEAlgo.h")

# Make an automatic library - will be static or dynamic based on user setting
add_library(CLUEAlgo_lib CLUEAlgo.cc ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(CLUEAlgo_lib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# This depends on (header only) boost
#target_link_libraries(CLUEAlgo_lib PRIVATE Boost::boost)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})

install(TARGETS CLUEAlgo_lib ${INSTALL_LIBRARIES}
  EXPORT CLUEAlgoTarget
  DESTINATION "${CMAKE_INSTALL_LIBDIR}")

# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.
#
add_executable(main main.cc)
target_compile_features(main PRIVATE cxx_std_17)

target_link_libraries(main PRIVATE CLUEAlgo_lib)

# mainCuplaCPUTBB: this will build the native C++ implementation of CLUE and
#       its corresponding TBB one  built using CUPLA. Which one to use must be
#       selected at runtime via a flag (misleadingly enough called useGPU in
#       the help text).
add_executable(mainCuplaCPUTBB main.cc)
target_compile_features(mainCuplaCPUTBB PRIVATE cxx_std_17)
target_compile_options(mainCuplaCPUTBB PRIVATE -DUSE_CUPLA -DFOR_TBB)

target_link_libraries(mainCuplaCPUTBB PRIVATE CLUEAlgo_lib tbb)


# mainCuplaCUDA: this will build the native C++ implementation of CLUE and its
#       corresponding CUDA one built using CUPLA. Which one to use must be
#       selected at runtime via a flag.



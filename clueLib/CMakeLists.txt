set(CMAKE_CXX_FLAGS "-fPIC")

## Make an automatic library - will be static or dynamic based on user setting
add_library(CLUE src/CLUEAlgo.cc)

# We need this directory, and users of our library will need it too
target_include_directories(CLUE PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clueLib/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#install(TARGETS CLUE ${INSTALL_LIBRARIES}
#  EXPORT CLUEAlgoTarget
#  DESTINATION "${CMAKE_INSTALL_LIBDIR}")

if(CMAKE_CUDA_COMPILER)
  add_library(CLUEGPU src/CLUEAlgoGPU.cu)

  target_compile_options(CLUEGPU PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler;
                        -m64;
                        -expt-relaxed-constexpr;
                        -w;
  >
  )

  target_include_directories(CLUEGPU PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clueLib/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#  install(TARGETS CLUEAlgoGPU_lib ${INSTALL_LIBRARIES}
#    EXPORT CLUEAlgoTarget
#    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif(CMAKE_CUDA_COMPILER)


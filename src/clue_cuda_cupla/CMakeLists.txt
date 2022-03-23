# mainCuplaCUDA: this will build the native C++ implementation of CLUE and its
#       corresponding CUDA one built using CUPLA. Which one to use must be
#       selected at runtime via a flag.

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CUDA)
  add_executable(mainCuplaCUDA ${PROJECT_SOURCE_DIR}/src/main.cc)
  target_compile_options(mainCuplaCUDA PRIVATE -DUSE_CUPLA -DFOR_CUDA -DALPAKA_ACC_GPU_CUDA_ENABLED=1)
  
  target_compile_options(mainCuplaCUDA PRIVATE -DUSE_EDM4HEP)
  target_link_libraries(mainCuplaCUDA PRIVATE CLUEAlgo_lib edm4hep podio::podioRootIO DD4hep::DDCore)

  target_include_directories(mainCuplaCUDA PUBLIC
    $<BUILD_INTERFACE:${cupla_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PUBLIC
    $<BUILD_INTERFACE:${cupla_DIR}/alpaka/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

  add_test(NAME mainCuplaCUDA_CPU 
           COMMAND mainCuplaCUDA ${TESTFILE_CSV} 20 25 2 0 1 1)
  set_tests_properties(mainCuplaCUDA_CPU PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

  add_test(NAME mainCuplaCUDA_GPU 
           COMMAND mainCuplaCUDA ${TESTFILE_CSV} 20 25 2 1 1 1)
  set_tests_properties(mainCuplaCUDA_GPU PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

endif(CMAKE_CUDA_COMPILER)
################################################################################

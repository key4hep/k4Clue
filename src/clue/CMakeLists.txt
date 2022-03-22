# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.

if(CMAKE_CUDA_COMPILER)
  # A copy with extension cu is needed to set the same source file in 
  # both languages (C++/CUDA)
  # configure_file(main.cc main.cu)
  
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CUDA)
  add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cc)
  
  target_compile_options(main PRIVATE -DUSE_EDM4HEP)
  target_link_libraries(main PRIVATE CLUEAlgo_lib CLUEAlgoGPU_lib edm4hep podio::podioRootIO DD4hep::DDCore)

  add_test(NAME main_CPU 
           COMMAND main /eos/project-k/key4hep/www/key4hep/testFiles/k4clue/inputData/standardCandles/aniso_1000.csv 20 25 2 0 1 1)
  set_tests_properties(main_CPU PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

  add_test(NAME main_GPU 
           COMMAND main /eos/project-k/key4hep/www/key4hep/testFiles/k4clue/inputData/standardCandles/aniso_1000.csv 20 25 2 1 1 1)
  set_tests_properties(main_GPU PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

endif(CMAKE_CUDA_COMPILER)

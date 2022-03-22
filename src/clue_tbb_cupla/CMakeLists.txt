# mainCuplaCPUTBB: this will build the native C++ implementation of CLUE and
#       its corresponding TBB one built using CUPLA. Which one to use must be
#       selected at runtime via a flag (misleadingly enough called useGPU in
#       the help text).
set(tbb_lib -ltbb)

set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CXX)
add_executable(mainCuplaCPUTBB ${PROJECT_SOURCE_DIR}/src/main.cc)

target_compile_options(mainCuplaCPUTBB PRIVATE -DUSE_CUPLA -DFOR_TBB ${tbb_lib} -DUSE_EDM4HEP)
target_link_libraries(mainCuplaCPUTBB PRIVATE CLUEAlgo_lib TBB::tbb EDM4HEP::edm4hep podio::podioRootIO DD4hep::DDCore)

target_include_directories(mainCuplaCPUTBB PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/alpaka/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${TBB_BASE}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_test(NAME mainCuplaCPUTBB_CPU 
         COMMAND mainCuplaCPUTBB /eos/project-k/key4hep/www/key4hep/testFiles/k4clue/inputData/standardCandles/aniso_1000.csv 20 25 2 0 1 1)
set_tests_properties(mainCuplaCPUTBB_CPU PROPERTIES
                     FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

add_test(NAME mainCuplaCPUTBB_TBB 
         COMMAND mainCuplaCPUTBB /eos/project-k/key4hep/www/key4hep/testFiles/k4clue/inputData/standardCandles/aniso_1000.csv 20 25 2 1 1 1)
set_tests_properties(mainCuplaCPUTBB_TBB PROPERTIES
                     FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

add_test(NAME mainEDM4HEP 
         COMMAND mainCuplaCPUTBB /eos/project-k/key4hep/www/key4hep/testFiles/k4clue/inputData/clic/20220322_gamma_10GeV_uniform_edm4hep.root 20 25 2 0 1 1)
set_tests_properties(mainEDM4HEP PROPERTIES
                     FAIL_REGULAR_EXPRESSION "Error;ERROR;Failed")

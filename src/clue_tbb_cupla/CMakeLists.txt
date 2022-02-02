# mainCuplaCPUTBB: this will build the native C++ implementation of CLUE and
#       its corresponding TBB one built using CUPLA. Which one to use must be
#       selected at runtime via a flag (misleadingly enough called useGPU in
#       the help text).
set(tbb_lib -ltbb)

set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CXX)
add_executable(mainCuplaCPUTBB ${PROJECT_SOURCE_DIR}/src/main.cc)

target_compile_options(mainCuplaCPUTBB PRIVATE -DUSE_CUPLA -DFOR_TBB ${tbb_lib} -DUSE_EDM4HEP)
target_link_libraries(mainCuplaCPUTBB PRIVATE CLUELib TBB::tbb EDM4HEP::edm4hep podio::podioRootIO DD4hep::DDCore)

target_include_directories(mainCuplaCPUTBB PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${cupla_DIR}/alpaka/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${TBB_BASE}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

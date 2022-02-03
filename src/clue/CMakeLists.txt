# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CUDA)
  add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cc)
  
  target_compile_options(main PRIVATE -DUSE_EDM4HEP  
  $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler;
                        -m64;
                        -expt-relaxed-constexpr;
                        -w;
  >
  )

  target_link_libraries(main PRIVATE CLUE CLUEGPU edm4hep podio::podioRootIO DD4hep::DDCore)

  target_include_directories(main PUBLIC
  #$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clueLib/include>
  #$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  #PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
endif(CMAKE_CUDA_COMPILER)

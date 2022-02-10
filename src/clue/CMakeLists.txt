# main: this will build the native C++ implementation of CLUE and its
#       corresponding native CUDA one. Which one to use must be selected at
#       runtime via a flag.

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${PROJECT_SOURCE_DIR}/src/main.cc PROPERTIES LANGUAGE CUDA)
  add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cc)
  
  target_compile_definitions(main PRIVATE USE_EDM4HEP)
  target_compile_options(main PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
                        -Xcompiler;
                        -m64;
                        -expt-relaxed-constexpr;
                        -w;
  >
  )

  target_link_libraries(main PRIVATE CLUE CLUEGPU EDM4HEP::edm4hep podio::podioRootIO DD4hep::DDCore)

  target_include_directories(main PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  )
endif(CMAKE_CUDA_COMPILER)

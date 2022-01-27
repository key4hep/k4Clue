#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CLUE" for configuration ""
set_property(TARGET CLUE APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(CLUE PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/libCLUE.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS CLUE )
list(APPEND _IMPORT_CHECK_FILES_FOR_CLUE "${_IMPORT_PREFIX}/lib64/libCLUE.a" )

# Import target "CLUEGPU" for configuration ""
set_property(TARGET CLUEGPU APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(CLUEGPU PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CUDA"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/libCLUEGPU.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS CLUEGPU )
list(APPEND _IMPORT_CHECK_FILES_FOR_CLUEGPU "${_IMPORT_PREFIX}/lib64/libCLUEGPU.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

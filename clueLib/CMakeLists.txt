## Make an automatic library - will be static or dynamic based on user setting
add_library(CLUELib src/CLUEAlgo.cc)

# We need this directory, and users of our library will need it too
target_include_directories(CLUELib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clueLib/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

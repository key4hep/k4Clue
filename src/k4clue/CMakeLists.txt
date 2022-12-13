# CLUE as Gaudi algorithm
find_package(k4FWCore)
find_package(EDM4HEP)

gaudi_add_module(ClueGaudiAlgorithmWrapper
  SOURCES
    ${PROJECT_SOURCE_DIR}/src/ClueGaudiAlgorithmWrapper.cpp
    ${PROJECT_SOURCE_DIR}/src/CLUECalorimeterHit.cpp
    ${PROJECT_SOURCE_DIR}/src/CLUENtuplizer.cpp
  LINK
    Gaudi::GaudiAlgLib
    Gaudi::GaudiKernel
    k4FWCore::k4FWCore
    DD4hep::DDCore
    EDM4HEP::edm4hep
    CLUEAlgo_lib
)

target_include_directories(ClueGaudiAlgorithmWrapper PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

add_test(NAME gaudiWrapper 
         WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
         COMMAND bash -c "source clueenv.sh; \
                 k4run ${PROJECT_SOURCE_DIR}/gaudi_opts/clue_gaudi_wrapper.py --ClueGaudiAlgorithmWrapperName.OutputLevel 2 --CLUEAnalysis.OutputLevel 2 --PandoraAnalysis.OutputLevel 2")

set_tests_properties(gaudiWrapper PROPERTIES
                     FAIL_REGULAR_EXPRESSION "ERROR;Failed")


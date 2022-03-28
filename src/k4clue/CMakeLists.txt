# CLUE as Gaudi algorithm
find_package(k4FWCore)
find_package(EDM4HEP)

gaudi_add_module(ClueGaudiAlgorithmWrapper
  SOURCES
    ${PROJECT_SOURCE_DIR}/src/ClueGaudiAlgorithmWrapper.cpp
  LINK
    Gaudi::GaudiAlgLib
    Gaudi::GaudiKernel
    k4FWCore::k4FWCore
    EDM4HEP::edm4hep
    CLUEAlgo_lib
)

target_include_directories(ClueGaudiAlgorithmWrapper PUBLIC
  k4FWCore::k4FWCore
  EDM4HEP::edm4hep
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

add_test(NAME gaudiWrapper 
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build/
         COMMAND bash -c "source clueenv.sh; \
                 k4run ../gaudi_opts/clue_gaudi_wrapper.py --ClueGaudiAlgorithmWrapperName.OutputLevel 2")

set(eventClustersExpr "ClueGaudiAlgorithmWrapper::execute")
set_property(TEST gaudiWrapper 
             PROPERTY PASS_REGULAR_EXPRESSION "${eventClustersExpr}")


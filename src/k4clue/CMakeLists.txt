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
    CLUE
)

target_include_directories(ClueGaudiAlgorithmWrapper PUBLIC
  k4FWCore::k4FWCore
  EDM4HEP::edm4hep
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

#[[
Copyright (c) 2020-2023 Key4hep-Project.

This file is part of Key4hep.
See https://key4hep.github.io/key4hep-doc/ for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

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
         COMMAND bash -c "source k4clueenv.sh; \
                 k4run ${PROJECT_SOURCE_DIR}/gaudi_opts/clue_gaudi_wrapper.py --ClueGaudiAlgorithmWrapperName.OutputLevel 2 --CLUEAnalysis.OutputLevel 2")

set_tests_properties(gaudiWrapper PROPERTIES
                     FAIL_REGULAR_EXPRESSION "ERROR;Failed")


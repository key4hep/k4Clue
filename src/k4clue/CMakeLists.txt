#[[
Copyright (c) 2020-2024 Key4hep-Project.

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

include(ExternalData)
set(ExternalData_URL_TEMPLATES
  "https://key4hep.web.cern.ch:443/testFiles/k4clue/inputData/clic/%(hash)"
)

gaudi_add_module(ClueGaudiAlgorithmWrapper
  SOURCES
    ${PROJECT_SOURCE_DIR}/src/ClueGaudiAlgorithmWrapper.cpp
    ${PROJECT_SOURCE_DIR}/src/CLUECalorimeterHit.cpp
    ${PROJECT_SOURCE_DIR}/src/CLUENtuplizer.cpp
  LINK
    Gaudi::GaudiKernel
    k4FWCore::k4FWCore
    DD4hep::DDCore
    EDM4HEP::edm4hep
    CLUEAlgo_lib
)

target_include_directories(ClueGaudiAlgorithmWrapper PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
ExternalData_Add_Test(k4clue_tests NAME gaudiWrapper
         WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
         COMMAND bash -c "k4run ${PROJECT_SOURCE_DIR}/gaudi_opts/clue_gaudi_wrapper.py --EventDataSvc.input DATA{${PROJECT_SOURCE_DIR}/test/input_files/20240905_gammaFromVertex_10GeV_uniform_10events_reco_edm4hep.root} --ClueGaudiAlgorithmWrapperName.OutputLevel 2 --CLUEAnalysis.OutputLevel 2")

set_tests_properties(gaudiWrapper PROPERTIES
                     FAIL_REGULAR_EXPRESSION "ERROR;Failed")

set_property(TEST gaudiWrapper APPEND PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}:${PROJECT_BINARY_DIR}/src/k4clue:$<$<TARGET_EXISTS:ROOT::Core>:$<TARGET_FILE_DIR:ROOT::Core>>:$<$<TARGET_EXISTS:EDM4HEP::edm4hep>:$<TARGET_FILE_DIR:EDM4HEP::edm4hep>>:$<$<TARGET_EXISTS:podio::podio>:$<TARGET_FILE_DIR:podio::podio>>:$ENV{LD_LIBRARY_PATH}")
set_property(TEST gaudiWrapper APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}/genConfDir:${PROJECT_BINARY_DIR}/src/k4clue/genConfDir:$ENV{PYTHONPATH}")        

ExternalData_Add_Target(k4clue_tests)

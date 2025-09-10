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

set(SOURCES_GPU
  ${PROJECT_SOURCE_DIR}/src/ClueGaudiAlgorithmWrapper.cpp
)

set(SOURCES_CPU
  ${PROJECT_SOURCE_DIR}/src/CLUECalorimeterHit.cpp
)

set(SOURCES ${SOURCES_CPU} ${SOURCES_GPU})

# ---------------------- GPU: HIP ----------------------
if(CMAKE_HIP_COMPILER)

  gaudi_add_module(ClueGaudiAlgorithmWrapperHIP
    SOURCES ${SOURCES}
    LINK Gaudi::GaudiKernel k4FWCore::k4FWCore DD4hep::DDCore EDM4HEP::edm4hep alpaka
  )

  target_include_directories(ClueGaudiAlgorithmWrapperHIP PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<$<COMPILE_LANGUAGE:HIP>:${hip_INCLUDE_DIRS}>
  )

  target_link_libraries(ClueGaudiAlgorithmWrapperHIP PUBLIC clue::CLUEstering)

  set_source_files_properties(${SOURCES_GPU} PROPERTIES LANGUAGE HIP)

  target_compile_options(ClueGaudiAlgorithmWrapperHIP PUBLIC
    $<$<COMPILE_LANGUAGE:HIP>:-DALPAKA_ACC_GPU_HIP_ENABLED>
    $<$<COMPILE_LANGUAGE:CXX>:-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND>
  )
endif()

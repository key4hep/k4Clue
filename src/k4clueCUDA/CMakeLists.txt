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

# ---------------------- GPU: CUDA ----------------------
if(CMAKE_CUDA_COMPILER)

  gaudi_add_module(ClueGaudiAlgorithmWrapperCUDA
    SOURCES ${SOURCES}
    LINK Gaudi::GaudiKernel k4FWCore::k4FWCore DD4hep::DDCore EDM4HEP::edm4hep alpaka
  )

  target_include_directories(ClueGaudiAlgorithmWrapperCUDA PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  )

  target_link_libraries(ClueGaudiAlgorithmWrapperCUDA PUBLIC clue::CLUEstering)

  set_target_properties(ClueGaudiAlgorithmWrapperCUDA PROPERTIES
    CUDA_STANDARD 20
    CXX_STANDARD 20
    CUDA_ARCHITECTURES 75
  )

  set_source_files_properties(${SOURCES_GPU} PROPERTIES LANGUAGE CUDA)

  target_compile_options(ClueGaudiAlgorithmWrapperCUDA PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-DALPAKA_ACC_GPU_CUDA_ENABLED --expt-relaxed-constexpr -gencode
    arch=compute_75,code=[sm_75,compute_75]>
  )

  target_link_libraries(ClueGaudiAlgorithmWrapperCUDA PUBLIC
    -L${CMAKE_CUDA_COMPILER}/../../lib64
  )
endif()

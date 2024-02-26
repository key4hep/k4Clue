#
# Copyright (c) 2020-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from Gaudi.Configuration import WARNING, DEBUG

from Configurables import k4DataSvc, MarlinProcessorWrapper

from Configurables import PodioInput
from Configurables import ClueGaudiAlgorithmWrapper
from Configurables import CLUENtuplizer
from Configurables import THistSvc
from Configurables import PodioOutput
from Configurables import ApplicationMgr

algList = []


evtsvc = k4DataSvc('EventDataSvc')
# evtsvc.input =

inp = PodioInput('InputReader')
inp.collections = [
  'EventHeader',
  'MCParticles',
  'ECALBarrel',
  'ECALEndcap',
]
inp.OutputLevel = WARNING

MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
MyAIDAProcessor.OutputLevel = WARNING
MyAIDAProcessor.ProcessorType = "AIDAProcessor"
MyAIDAProcessor.Parameters = {"FileName": ["histograms_clue_standalone"],
                    "FileType": ["root"],
                    "Compress": ["1"],
                    }



MyClueGaudiAlgorithmWrapper = ClueGaudiAlgorithmWrapper("ClueGaudiAlgorithmWrapperName")
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "ECALBarrel"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "ECALEndcap"
MyClueGaudiAlgorithmWrapper.CriticalDistance = 15.00
MyClueGaudiAlgorithmWrapper.MinLocalDensity = 0.02
MyClueGaudiAlgorithmWrapper.OutlierDeltaFactor = 3.00
MyClueGaudiAlgorithmWrapper.OutputLevel = DEBUG

MyCLUENtuplizer = CLUENtuplizer("CLUEAnalysis")
MyCLUENtuplizer.ClusterCollection = "CLUEClusters"
MyCLUENtuplizer.BarrelCaloHitsCollection = "ECALBarrel"
MyCLUENtuplizer.EndcapCaloHitsCollection = "ECALEndcap"
MyCLUENtuplizer.SingleMCParticle = True
MyCLUENtuplizer.OutputLevel = WARNING

THistSvc().Output = ["rec DATAFILE='k4clue_analysis_output.root' TYP='ROOT' OPT='RECREATE'"]
THistSvc().OutputLevel = WARNING
THistSvc().PrintAll = False
THistSvc().AutoSave = True
THistSvc().AutoFlush = True

out = PodioOutput("out")
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "ECALBarrel"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "ECALEndcap"
out.filename = "my_output_clue_standalone.root"
out.outputCommands = ["keep *"]

algList.append(inp)
algList.append(MyAIDAProcessor)
algList.append(MyClueGaudiAlgorithmWrapper)
algList.append(MyCLUENtuplizer)
algList.append(out)

ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax   = 3,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

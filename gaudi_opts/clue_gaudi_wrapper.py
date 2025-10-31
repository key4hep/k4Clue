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
from Configurables import ClueGaudiAlgorithmWrapper3D, CLUENtuplizer, THistSvc, EventDataSvc
from k4FWCore import ApplicationMgr, IOSvc

iosvc = IOSvc()
# iosvc.Input =
iosvc.Output = "clueReco.root"

dc = 30
rho = 0.1
dm = 120

MyClueGaudiAlgorithmWrapper = ClueGaudiAlgorithmWrapper3D("ClueGaudiAlgorithmWrapperName",
    BarrelCaloHitsCollection = ["ECALBarrel"],
    EndcapCaloHitsCollection = ["ECALEndcap"],
    CriticalDistance = dc,
    MinLocalDensity = rho,
    FollowerDistance = dm,
    OutputLevel = DEBUG
)

MyCLUENtuplizer = CLUENtuplizer("CLUEAnalysis",
    ClusterCollection = "CLUEClusters",
    BarrelCaloHitsCollection = ["ECALBarrel"],
    EndcapCaloHitsCollection = ["ECALEndcap"],
    OutputLevel = WARNING
)

str_params = str(rho).replace(".","p") + "_" + str(dc).replace(".","p") + "_" + str(dm).replace(".","p")
filename = "rec DATAFILE='k4clue_analysis_output_3D_"+str_params+".root' TYP='ROOT' OPT='RECREATE'"
THistSvc().Output = [filename]
THistSvc().OutputLevel = WARNING
THistSvc().PrintAll = False
THistSvc().AutoSave = True
THistSvc().AutoFlush = True

ApplicationMgr( TopAlg = [MyClueGaudiAlgorithmWrapper, MyCLUENtuplizer],
                EvtSel = 'NONE',
                EvtMax   = 3,
                ExtSvc = [EventDataSvc("EventDataSvc")],
                OutputLevel=WARNING
              )

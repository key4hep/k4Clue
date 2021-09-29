from Gaudi.Configuration import *

from Configurables import LcioEvent, k4DataSvc
from k4MarlinWrapper.parseConstants import *
algList = []

from Configurables import PodioInput
evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = '../data/input/clic/gamma_energy_10GeV_theta_70deg_90deg.root'
#evtsvc.input = '../data/input/clic/ttbar_3ev.root'

inp = PodioInput('InputReader')
inp.collections = [
  'EB_CaloHits_EDM4hep',
  'EE_CaloHits_EDM4hep',
  'PandoraClusters_EDM4hep',
]
inp.OutputLevel = DEBUG

from Configurables import MarlinProcessorWrapper

MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
MyAIDAProcessor.OutputLevel = WARNING
MyAIDAProcessor.ProcessorType = "AIDAProcessor"
MyAIDAProcessor.Parameters = {"FileName": ["histograms"],
                    "FileType": ["root"],
                    "Compress": ["1"],
                    }


from Configurables import ClueGaudiAlgorithmWrapper

MyClueGaudiAlgorithmWrapper = ClueGaudiAlgorithmWrapper("ClueGaudiAlgorithmWrapperName")
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "EB_CaloHits_EDM4hep"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "EE_CaloHits_EDM4hep"
MyClueGaudiAlgorithmWrapper.CriticalDistance = 10.00
MyClueGaudiAlgorithmWrapper.MinLocalDensity = 0.02
MyClueGaudiAlgorithmWrapper.OutlierDeltaFactor = 1.00

from Configurables import PodioOutput
out = PodioOutput("out")
out.filename = "output.root"
out.outputCommands = ["keep *"]

algList.append(inp)
algList.append(MyAIDAProcessor)
algList.append(MyClueGaudiAlgorithmWrapper)
algList.append(out)

from Configurables import ApplicationMgr
ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax   = -1,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

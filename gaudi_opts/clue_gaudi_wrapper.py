from Gaudi.Configuration import *

from Configurables import LcioEvent, k4DataSvc
from k4MarlinWrapper.parseConstants import *
algList = []

from Configurables import PodioInput
evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = 'https://key4hep.web.cern.ch/testFiles/k4clue/inputData/clic/20220322_gamma_10GeV_uniform_edm4hep.root'

inp = PodioInput('InputReader')
inp.collections = [
  'ECALBarrel',
  'ECALEndcap'
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
MyClueGaudiAlgorithmWrapper.OutputLevel = WARNING
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "ECALBarrel"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "ECALEndcap"
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
                EvtMax   = 3,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

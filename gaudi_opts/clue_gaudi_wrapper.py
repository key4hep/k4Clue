from Gaudi.Configuration import *

from Configurables import LcioEvent, k4DataSvc
from k4MarlinWrapper.parseConstants import *
algList = []

from Configurables import PodioInput
evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = '../data/input/clic_gamma_barrel.root'

inp = PodioInput('InputReader')
inp.collections = [
  'EB_CaloHits_EDM4hep',
  'EE_CaloHits_EDM4hep',
]
inp.OutputLevel = DEBUG

END_TAG = "END_TAG"

from Configurables import MarlinProcessorWrapper

MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
MyAIDAProcessor.OutputLevel = WARNING
MyAIDAProcessor.ProcessorType = "AIDAProcessor"
MyAIDAProcessor.Parameters = ["FileName", "histograms", END_TAG,
                    "FileType", "root", END_TAG,
                    "Compress", "1", END_TAG,
                    ]


from Configurables import ClueGaudiAlgorithmWrapper

MyClueGaudiAlgorithmWrapper = ClueGaudiAlgorithmWrapper("ClueGaudiAlgorithmWrapperName")
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
                EvtMax   = 2,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

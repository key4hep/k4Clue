from Gaudi.Configuration import *

from Configurables import LcioEvent, k4DataSvc
from k4MarlinWrapper.parseConstants import *
algList = []

from Configurables import PodioInput
evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = '../data/input/clic/gamma_energy_10GeV_theta_10deg_30deg.root'


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

algList.append(inp)
algList.append(MyAIDAProcessor)
algList.append(MyClueGaudiAlgorithmWrapper)


from Configurables import ApplicationMgr
ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax   = 1,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

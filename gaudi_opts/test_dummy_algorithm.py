from Gaudi.Configuration import *

from Configurables import LcioEvent, k4DataSvc
from k4MarlinWrapper.parseConstants import *
algList = []

from Configurables import PodioInput
evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = '../ttbar3_edm4hep.root'


inp = PodioInput('InputReader')
inp.collections = [
  # 'HcalBarrelHits',
  'EventHeader',
  'VertexBarrelCollection',
  'VertexEndcapCollection',
  'InnerTrackerBarrelCollection',
  'OuterTrackerBarrelCollection',
  'ECalEndcapCollection',
  'ECalEndcapCollectionContributions',
  'ECalBarrelCollection',
  'ECalPlugCollection',
  'HCalBarrelCollection',
  'HCalBarrelCollectionContributions',
  'InnerTrackerEndcapCollection',
  'OuterTrackerEndcapCollection',
  'HCalEndcapCollection',
  'HCalEndcapCollectionContributions',
  'HCalRingCollection',
  'HCalRingCollectionContributions',
  'YokeEndcapCollection',
  'YokeEndcapCollectionContributions',
  'LumiCalCollection',
  'LumiCalCollectionContributions'
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


from Configurables import DummyAlgorithm

MyDummyAlgorithm = DummyAlgorithm("DummyAlgorithmName")

algList.append(inp)
algList.append(MyAIDAProcessor)
algList.append(MyDummyAlgorithm)


from Configurables import ApplicationMgr
ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax   = 1,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

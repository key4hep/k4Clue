from Gaudi.Configuration import *

from Configurables import LcioEvent, k4DataSvc, MarlinProcessorWrapper
from k4MarlinWrapper.parseConstants import *
algList = []

from Configurables import PodioInput
evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = 'https://key4hep.web.cern.ch/testFiles/k4clue/inputData/clic/20230825_gammaFromSurface_10GeV_uniform_500events_reco_edm4hep.root'

inp = PodioInput('InputReader')
inp.collections = [
  'EventHeader',
  'MCParticles',
  'ECALBarrel',
  'ECALEndcap',
]
inp.OutputLevel = DEBUG

MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
MyAIDAProcessor.OutputLevel = WARNING
MyAIDAProcessor.ProcessorType = "AIDAProcessor"
MyAIDAProcessor.Parameters = {"FileName": ["histograms_clue_standalone"],
                    "FileType": ["root"],
                    "Compress": ["1"],
                    }


from Configurables import ClueGaudiAlgorithmWrapper

MyClueGaudiAlgorithmWrapper = ClueGaudiAlgorithmWrapper("ClueGaudiAlgorithmWrapperName")
MyClueGaudiAlgorithmWrapper.OutputLevel = DEBUG
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "ECALBarrel"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "ECALEndcap"
MyClueGaudiAlgorithmWrapper.CriticalDistance = 15.00
MyClueGaudiAlgorithmWrapper.MinLocalDensity = 0.02
MyClueGaudiAlgorithmWrapper.OutlierDeltaFactor = 3.00

from Configurables import CLUENtuplizer
MyCLUENtuplizer = CLUENtuplizer("CLUEAnalysis")
MyCLUENtuplizer.ClusterCollection = "CLUEClusters"
MyCLUENtuplizer.BarrelCaloHitsCollection = "ECALBarrel"
MyCLUENtuplizer.EndcapCaloHitsCollection = "ECALEndcap"
MyCLUENtuplizer.SingleMCParticle = True
MyCLUENtuplizer.OutputLevel = WARNING

from Configurables import THistSvc
THistSvc().Output = ["rec DATAFILE='k4clue_analysis_output.root' TYP='ROOT' OPT='RECREATE'"]
THistSvc().OutputLevel = WARNING
THistSvc().PrintAll = False
THistSvc().AutoSave = True
THistSvc().AutoFlush = True

from Configurables import PodioOutput
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

from Configurables import ApplicationMgr
ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax   = 3,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )

#include "ClueGaudiAlgorithmWrapper.h"

#include "IO_helper.h"
#include "CLUEAlgo.h"

DECLARE_COMPONENT(ClueGaudiAlgorithmWrapper)

ClueGaudiAlgorithmWrapper::ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* pSL) :
  GaudiAlgorithm(name, pSL), m_eventDataSvc("EventDataSvc", "ClueGaudiAlgorithmWrapper") {
  declareProperty("BarrelCaloHitsCollection", EBCaloCollectionName, "Collection for Barrel Calo Hits used in input");
  declareProperty("EndcapCaloHitsCollection", EECaloCollectionName, "Collection for Endcap Calo Hits used in input");
  declareProperty("CriticalDistance", dc, "Used to compute the local density");
  declareProperty("MinLocalDensity", rhoc, "Minimum local density for a point to be promoted as a Seed");
  declareProperty("OutlierDeltaFactor", outlierDeltaFactor, "Multiplicative constant to be applied to CriticalDistance");
  declareProperty("OutClusters", clustersHandle, "Clusters collection (output)");
  declareProperty("OutCaloHits", caloHitsHandle, "Calo hits collection created from Clusters (output)");

  StatusCode sc = m_eventDataSvc.retrieve();
}

StatusCode ClueGaudiAlgorithmWrapper::initialize() {
  debug() << "ClueGaudiAlgorithmWrapper::initialize()" << endmsg ;

  m_podioDataSvc = dynamic_cast<PodioDataSvc*>(m_eventDataSvc.get());
  if (m_podioDataSvc == nullptr) {
    return StatusCode::FAILURE;
  }

  return Algorithm::initialize();
}

std::map<int, std::vector<int> > ClueGaudiAlgorithmWrapper::runAlgo( std::vector<float>& x, std::vector<float>& y, 
                                                                     std::vector<int>& layer, std::vector<float>& weight ){

  // Run CLUE
  debug() << "Using CLUEAlgo ... " << endmsg;
  CLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, false);
  clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
  // measure excution time of makeClusters
  auto start = std::chrono::high_resolution_clock::now();
  clueAlgo.makeClusters();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  debug() << "Elapsed time: " << elapsed.count() *1000 << " ms" << endmsg ;
  // output result to outputFileName. -1 means all points.

  debug() << "Finished running CLUE algorithm" << endmsg;

  std::map<int, std::vector<int> > clueClusters = clueAlgo.getClusters();
  return clueClusters;
}

StatusCode ClueGaudiAlgorithmWrapper::execute() {
  debug() << "ClueGaudiAlgorithmWrapper::execute()" << endmsg ;

  // Read data and run algo
  DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle {  
    EBCaloCollectionName, Gaudi::DataHandle::Reader, this};

  const auto EB_calo_coll = EB_calo_handle.get();

  if( EB_calo_coll->isValid() ) {
    for(const auto& calo_hit_EB : (*EB_calo_coll) ){
      calo_coll->push_back(calo_hit_EB.clone());
    }
  } else {
    throw std::runtime_error("Collection not found.");
  }
  info() << EB_calo_coll->size() << " caloHits in " << EBCaloCollectionName << "." << endmsg;

  // Get collection metadata cellID which is valid for both EB and EE
  auto EB_collID = EB_calo_coll->getID();
  const auto cellIDstr = EB_calo_handle.getCollMetadataCellID(EB_collID);

  DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle {  
    EECaloCollectionName, Gaudi::DataHandle::Reader, this};

  const auto EE_calo_coll = EE_calo_handle.get();

  if( EE_calo_coll->isValid() ) {
    for(const auto& calo_hit_EE : (*EE_calo_coll) ){
      calo_coll->push_back(calo_hit_EE.clone());
    }
  } else {
    throw std::runtime_error("Collection not found.");
  }
  info() << EE_calo_coll->size() << " caloHits in " << EECaloCollectionName << "." << endmsg;

  debug() << calo_coll->size() << " caloHits in total. " << endmsg;
  read_EDM4HEP_event(calo_coll, cellIDstr, x, y, layer, weight);

  std::map<int, std::vector<int> > clueClusters = runAlgo(x, y, layer, weight);
  debug() << "Produced " << clueClusters.size() << " clusters" << endmsg;

  // Save CLUECaloHits
  clue::CLUECalorimeterHitCollection clue_coll;
  for(auto ch : calo_coll) {
    info() << "CH      : " << ch.getPosition().x << endmsg;
    clue::CLUECalorimeterHit cluech(ch.clone(), 100, 1, 1, 3.0, 1.2);
    clue_coll.vect.push_back(cluech);
    info() << "CH CLUE layer, pos : " << cluech.getLayer() << " " << cluech.getPosition().x << endmsg;
    info() << endmsg;
  }
  info() << "-> clue coll final size " << clue_coll.vect.size() << endmsg;

  auto pCHV = std::make_unique<clue::CLUECalorimeterHitCollection>(clue_coll);
  const StatusCode scStatusV = eventSvc()->registerObject("/Event/CLUECalorimeterHitCollection", pCHV.release());

  // Save clusters
  edm4hep::ClusterCollection* finalClusters = clustersHandle.createAndPut();
  computeClusters(calo_coll, cellIDstr, EB_calo_coll, EE_calo_coll, clueClusters, finalClusters);
  info() << "Saved " << finalClusters->size() << " clusters" << endmsg;

  // Save clusters as calo hits
  edm4hep::CalorimeterHitCollection* finalCaloHits = caloHitsHandle.createAndPut();
  computeCaloHits(calo_coll, cellIDstr, clueClusters, finalCaloHits);
  // Add cellID to calohits
  auto& calohits_md = m_podioDataSvc->getProvider().getCollectionMetaData(finalCaloHits->getID());
  calohits_md.setValue("CellIDEncodingString", cellIDstr);

  debug() << "Saved " << finalCaloHits->size() << " clusters as calo hits" << endmsg;

  //Cleaning
  calo_coll.clear();
  x.clear();
  y.clear();
  layer.clear();
  weight.clear();
 
  return StatusCode::SUCCESS;
}

StatusCode ClueGaudiAlgorithmWrapper::finalize() {
  debug() << "ClueGaudiAlgorithmWrapper::finalize()" << endmsg ;
  return Algorithm::finalize();
}

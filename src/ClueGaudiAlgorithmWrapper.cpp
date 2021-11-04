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
  declareProperty("OutClustersFake", fakeClustersHandle, "Fake clusters collection (output)");

  StatusCode sc = m_eventDataSvc.retrieve();
}

StatusCode ClueGaudiAlgorithmWrapper::initialize() {
  std::cout << "ClueGaudiAlgorithmWrapper::initialize()\n";

  m_podioDataSvc = dynamic_cast<PodioDataSvc*>(m_eventDataSvc.get());
  if (m_podioDataSvc == nullptr) {
    return StatusCode::FAILURE;
  }

  return Algorithm::initialize();
}

std::map<int, std::vector<int> > ClueGaudiAlgorithmWrapper::runAlgo( std::vector<float>& x, std::vector<float>& y, 
                                                                     std::vector<int>& layer, std::vector<float>& weight ){

  // Run CLUE
  std::cout << "Using CLUEAlgo ... " << std::endl;
  CLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, false);
  clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
  // measure excution time of makeClusters
  auto start = std::chrono::high_resolution_clock::now();
  clueAlgo.makeClusters();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() *1000 << " ms\n";
  // output result to outputFileName. -1 means all points.

  std::cout << "Finished running CLUE algorithm" << std::endl;

  std::map<int, std::vector<int> > clueClusters = clueAlgo.getClusters();
  return clueClusters;
}

StatusCode ClueGaudiAlgorithmWrapper::execute() {
  std::cout << "ClueGaudiAlgorithmWrapper::execute()\n";

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
  std::cout << EB_calo_coll->size() << " caloHits in " << EBCaloCollectionName << "." << std::endl;

  // Get collection metadata cellID
  auto EB_collID = EB_calo_coll->getID();
  const auto EB_cellIDstr = EB_calo_handle.getCollMetadataCellID(EB_collID);

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
  std::cout << EE_calo_coll->size() << " caloHits in " << EBCaloCollectionName << "." << std::endl;

  // Get collection metadata cellID
  auto EE_collID = EE_calo_coll->getID();
  const auto EE_cellIDstr = EE_calo_handle.getCollMetadataCellID(EE_collID);

  std::cout << calo_coll->size() << " caloHits in total. " << std::endl;
  read_EDM4HEP_event(calo_coll, x, y, layer, weight);

  std::map<int, std::vector<int> > clueClusters = runAlgo(x, y, layer, weight);
  std::cout << "Produced " << clueClusters.size() << " clusters" << std::endl;

  // Save clusters
  edm4hep::ClusterCollection* finalClusters = clustersHandle.createAndPut();
  computeClusters(calo_coll, clueClusters, finalClusters);
  std::cout << "Saved " << finalClusters->size() << " clusters" << std::endl;

  edm4hep::CalorimeterHitCollection* finalCaloHits = fakeClustersHandle.createAndPut();
  computeCaloHits(calo_coll, clueClusters, finalCaloHits);
  // Add cellID to calohits
  auto& callohits_md = m_podioDataSvc->getProvider().getCollectionMetaData(finalCaloHits->getID());
  callohits_md.setValue("CellIDEncodingString", EB_cellIDstr);

  std::cout << "Saved " << finalCaloHits->size() << " clusters as calo hits" << std::endl;

  //Cleaning
  calo_coll.clear();
  x.clear();
  y.clear();
  layer.clear();
  weight.clear();
 
  return StatusCode::SUCCESS;
}

StatusCode ClueGaudiAlgorithmWrapper::finalize() {
  std::cout << "ClueGaudiAlgorithmWrapper::finalize()\n";
  return Algorithm::finalize();
}

#include "ClueGaudiAlgorithmWrapper.h"

#include "IO_helper.h"
#include "CLUEAlgo.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

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

void ClueGaudiAlgorithmWrapper::fillInputs(){

  for (const auto& ch : clue_hit_coll.vect) {
    x.push_back(ch.getEta());
    y.push_back(ch.getPhi());
    layer.push_back(ch.getLayer());
    weight.push_back(ch.getEnergy());
  }
  return;
}

std::map<int, std::vector<int> > ClueGaudiAlgorithmWrapper::runAlgo(){

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
  Points cluePoints = clueAlgo.getPoints();

  for(int i = 0; i < cluePoints.n; i++){

    clue_hit_coll.vect[i].setRho(cluePoints.rho[i]);
    clue_hit_coll.vect[i].setDelta(cluePoints.delta[i]);
    clue_hit_coll.vect[i].setClusterIndex(cluePoints.clusterIndex[i]);

    if(cluePoints.isSeed[i] == 1){
      clue_hit_coll.vect[i].setStatus(clue::CLUECalorimeterHit::Status::seed);
    } else if (cluePoints.clusterIndex[i] == -1) {
      clue_hit_coll.vect[i].setStatus(clue::CLUECalorimeterHit::Status::outlier);
    } else {
      clue_hit_coll.vect[i].setStatus(clue::CLUECalorimeterHit::Status::follower);
    }

  }
  return clueClusters;
}

void ClueGaudiAlgorithmWrapper::fillFinalClusters(const std::map<int, std::vector<int> > clusterMap, 
                                                  edm4hep::ClusterCollection* clusters){

  for(auto cl : clusterMap){

    info() << cl.first << " with size " << cl.second.size() << endmsg;

    // Outliers should not create a cluster
    if(cl.first == -1){
      continue;
    }

    std::map<int, std::vector<int> > clustersLayer;
    for(auto index : cl.second){
      clustersLayer[clue_hit_coll.vect.at(index).getLayer()].push_back(index);
    }

    for(auto clLay : clustersLayer){
      auto position = edm4hep::Vector3f({0,0,0});

      auto cluster = clusters->create();
      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;
      //info() << "  layer = " << clLay.first << endmsg;
      for(auto index : clLay.second){
        //info() << "    " << index << endmsg;

        position.x += clue_hit_coll.vect.at(index).getPosition().x;
        position.y += clue_hit_coll.vect.at(index).getPosition().y;
        position.z += clue_hit_coll.vect.at(index).getPosition().z;
        if( EB_calo_coll->size() != 0){
          if( index < EB_calo_coll->size() ) {
            cluster.addToHits(EB_calo_coll->at(index));
          } else {
            cluster.addToHits(EE_calo_coll->at(index - EB_calo_coll->size()));
          }
        } else {
          cluster.addToHits(EE_calo_coll->at(index));
        }

        if (clue_hit_coll.vect.at(index).getEnergy() > maxEnergyValue) {
          maxEnergyValue = clue_hit_coll.vect.at(index).getEnergy();
          maxEnergyIndex = index;
        }
      }
      float energy = 0.f;
      float sumEnergyErrSquared = 0.f;
      std::for_each(cluster.getHits().begin(), cluster.getHits().end(),
                    [&energy, &sumEnergyErrSquared] (edm4hep::CalorimeterHit elem) { 
                      energy += elem.getEnergy(); 
                      sumEnergyErrSquared += pow(elem.getEnergyError()/(1.*elem.getEnergy()), 2);
                    });
      cluster.setEnergy(energy);
      cluster.setEnergyError(sqrt(sumEnergyErrSquared));

      calculatePosition(&cluster);

      //JUST A PLACEHOLDER FOR NOW: TO BE FIXED
      cluster.setPositionError({0.00, 0.00, 0.00, 0.00, 0.00, 0.00});
      cluster.setType(clue_hit_coll.vect.at(maxEnergyIndex).getType());
    }
    clustersLayer.clear();
  }
  info() << "Total num clusters = " << clusters->size() << endmsg;
  return;
}

void ClueGaudiAlgorithmWrapper::calculatePosition(edm4hep::MutableCluster* cluster) {

  float total_weight = cluster->getEnergy();
  if(total_weight <= 0)
    warning() << "Zero energy in the cluster" << endmsg;

  float total_weight_log = 0.f;
  float x_log = 0.f;
  float y_log = 0.f;
  float z_log = 0.f;
  double thresholdW0_ = 2.9; //Min percentage of energy to contribute to the log-reweight position

  float maxEnergyValue = 0.f;
  unsigned int maxEnergyIndex = 0;
  for (int i = 0; i < cluster->hits_size(); i++) {
    float rhEnergy = cluster->getHits(i).getEnergy();
    float Wi = std::max(thresholdW0_ - std::log(rhEnergy / total_weight), 0.);
    x_log += cluster->getHits(i).getPosition().x * Wi;
    y_log += cluster->getHits(i).getPosition().y * Wi;
    z_log += cluster->getHits(i).getPosition().z * Wi;
    total_weight_log += Wi;
  }

  if (total_weight_log != 0.) {
    float inv_tot_weight_log = 1.f / total_weight_log;
    cluster->setPosition({x_log * inv_tot_weight_log, y_log * inv_tot_weight_log, z_log * inv_tot_weight_log});
  }

  return;
}

void ClueGaudiAlgorithmWrapper::transformClustersInCaloHits(edm4hep::ClusterCollection* clusters,
                                                            edm4hep::CalorimeterHitCollection* caloHits){

  float time = 0.f;
  float maxEnergy = 0.f;
  std::uint64_t maxEnergyCellID = 0;

  for(auto cl : *clusters){
    auto caloHit = caloHits->create();
    caloHit.setEnergy(cl.getEnergy());
    caloHit.setEnergyError(cl.getEnergyError());
    caloHit.setPosition(cl.getPosition());
    caloHit.setType(cl.getType());

    time = 0.0;
    maxEnergy = 0.0;
    maxEnergyCellID = 0;
    for(auto hit : cl.getHits()){
      time += hit.getTime();
      if (hit.getEnergy() > maxEnergy) {
        maxEnergy = hit.getEnergy();
        maxEnergyCellID = hit.getCellID();
      }
    }
    caloHit.setCellID(maxEnergyCellID);
    caloHit.setTime(time/cl.hits_size());
  }

  return;
}

StatusCode ClueGaudiAlgorithmWrapper::execute() {
  debug() << "ClueGaudiAlgorithmWrapper::execute()" << endmsg ;

  // Read EB collection
  DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle {  
    EBCaloCollectionName, Gaudi::DataHandle::Reader, this};
  EB_calo_coll = EB_calo_handle.get();

  // Read EE collection
  DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle {  
    EECaloCollectionName, Gaudi::DataHandle::Reader, this};
  EE_calo_coll = EE_calo_handle.get();

  // Get collection metadata cellID which is valid for both EB and EE
  auto collID = EB_calo_coll->getID();
  const auto cellIDstr = EB_calo_handle.getCollMetadataCellID(collID);
  const BitFieldCoder bf(cellIDstr);

  // Fill CLUECaloHits
  if( EB_calo_coll->isValid() ) {
    for(const auto& calo_hit : (*EB_calo_coll) ){
      clue_hit_coll.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::barrel, bf.get( calo_hit.getCellID(), "layer")));
    }
  } else {
    throw std::runtime_error("Collection not found.");
  }
  info() << EB_calo_coll->size() << " caloHits in " << EBCaloCollectionName << "." << endmsg;

  if( EE_calo_coll->isValid() ) {
    for(const auto& calo_hit : (*EE_calo_coll) ){
      clue_hit_coll.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::endcap, bf.get( calo_hit.getCellID(), "layer")));
    }
  } else {
    throw std::runtime_error("Collection not found.");
  }
  info() << EE_calo_coll->size() << " caloHits in " << EECaloCollectionName << "." << endmsg;

  debug() << clue_hit_coll.vect.size() << " caloHits in total. " << endmsg;

  // Fill CLUECaloHits
  fillInputs();

  std::map<int, std::vector<int> > clueClusters = runAlgo();
  debug() << "Produced " << clueClusters.size() << " clusters" << endmsg;

  auto pCHV = std::make_unique<clue::CLUECalorimeterHitCollection>(clue_hit_coll);
  const StatusCode scStatusV = eventSvc()->registerObject("/Event/CLUECalorimeterHitCollection", pCHV.release());

  // Save clusters
  edm4hep::ClusterCollection* finalClusters = clustersHandle.createAndPut();
  fillFinalClusters(clueClusters, finalClusters);
  info() << "Saved " << finalClusters->size() << " clusters" << endmsg;

  // Add cellID to clusters
  auto& clusters_md = m_podioDataSvc->getProvider().getCollectionMetaData(finalClusters->getID());
  clusters_md.setValue("CellIDEncodingString", cellIDstr);

  // Save clusters as calo hits
  edm4hep::CalorimeterHitCollection* finalCaloHits = caloHitsHandle.createAndPut();
  transformClustersInCaloHits(finalClusters, finalCaloHits);
  debug() << "Saved " << finalCaloHits->size() << " clusters as calo hits" << endmsg;

  // Add cellID to calohits
  auto& calohits_md = m_podioDataSvc->getProvider().getCollectionMetaData(finalCaloHits->getID());
  calohits_md.setValue("CellIDEncodingString", cellIDstr);

  //Cleaning
  clue_hit_coll.vect.clear();
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

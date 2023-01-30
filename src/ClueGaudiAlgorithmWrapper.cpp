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

  m_podioDataSvc = dynamic_cast<PodioDataSvc*>(m_eventDataSvc.get());
  if (m_podioDataSvc == nullptr) {
    return StatusCode::FAILURE;
  }

  return Algorithm::initialize();

}

void ClueGaudiAlgorithmWrapper::fillCLUEPoints(std::vector<clue::CLUECalorimeterHit>& clue_hits){

  for (const auto& ch : clue_hits) {
    if(ch.inBarrel()){
      x.push_back(ch.getPhi()*ch.getR());
      y.push_back(ch.getPosition().z);
      r.push_back(ch.getR());
    } else {
      x.push_back(ch.getPosition().x);
      y.push_back(ch.getPosition().y);
      // For the endcap the r info is not mandatory because it is not used
      r.push_back(ch.getR());
    }
    layer.push_back(ch.getLayer());
    weight.push_back(ch.getEnergy());
  }
  return;

}

std::map<int, std::vector<int> > ClueGaudiAlgorithmWrapper::runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits, 
								    bool isBarrel = false){

  std::map<int, std::vector<int> > clueClusters;
  Points cluePoints;

  // Fill CLUE inputs
  fillCLUEPoints(clue_hits);

  // Run CLUE
  info() << "Running CLUEAlgo ... " << endmsg;
  if(isBarrel){
    info() << "... in the barrel" << endmsg;
    CLDBarrelCLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, true);
    if(clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0], &r[0]))
      throw error() << "Error in setting the clue points for the barrel." << endmsg;
    clueAlgo.makeClusters();
    clueClusters = clueAlgo.getClusters();
    cluePoints = clueAlgo.getPoints();
  } else {
    info() << "... in the endcap" << endmsg;
    CLDEndcapCLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, true);
    if(clueAlgo.setPoints(x.size(), &x[0], &y[0], &layer[0], &weight[0], &r[0]))
      throw error() << "Error in setting the clue points for the endcap." << endmsg;
    clueAlgo.makeClusters();
    clueClusters = clueAlgo.getClusters();
    cluePoints = clueAlgo.getPoints();
  }

  info() << "Finished running CLUE algorithm" << endmsg;

  // Including CLUE info in cluePoints
  for(int i = 0; i < cluePoints.n; i++){

    clue_hits[i].setRho(cluePoints.rho[i]);
    clue_hits[i].setDelta(cluePoints.delta[i]);
    clue_hits[i].setClusterIndex(cluePoints.clusterIndex[i]);
    debug() << "CLUE Point #" << i <<" : (x,y,z) = (" 
           << clue_hits[i].getPosition().x << ","
           << clue_hits[i].getPosition().y << ","
           << clue_hits[i].getPosition().z << ")";

    if(cluePoints.isSeed[i] == 1){
      debug() << " is seed" << endmsg; 
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::seed);
    } else if (cluePoints.clusterIndex[i] == -1) {
      debug() << " is outlier" << endmsg; 
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::outlier);
    } else {
      debug() << " is follower of cluster #" << cluePoints.clusterIndex[i] << endmsg; 
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::follower);
    }

  }

  // Clean CLUE inputs
  cleanCLUEPoints();

  return clueClusters;
}

void ClueGaudiAlgorithmWrapper::cleanCLUEPoints(){
  x.clear();
  y.clear();
  r.clear();
  layer.clear();
  weight.clear();
}

void ClueGaudiAlgorithmWrapper::fillFinalClusters(std::vector<clue::CLUECalorimeterHit>& clue_hits, 
                                                  const std::map<int, std::vector<int> > clusterMap, 
                                                  edm4hep::ClusterCollection* clusters){

  for(auto cl : clusterMap){

    // Outliers should not create a cluster
    if(cl.first == -1){
      continue;
    }

    std::map<int, std::vector<int> > clustersLayer;
    for(auto index : cl.second){
      clustersLayer[clue_hits[index].getLayer()].push_back(index);
    }

    for(auto clLay : clustersLayer){

      auto cluster = clusters->create();
      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;

      for(auto index : clLay.second){

        if(clue_hits[index].inBarrel()){
          cluster.addToHits(EB_calo_coll->at(index));
        }
        if(clue_hits[index].inEndcap()){
          cluster.addToHits(EE_calo_coll->at(index));
        }
        
        if (clue_hits[index].getEnergy() > maxEnergyValue) {
          maxEnergyValue = clue_hits[index].getEnergy();
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
      cluster.setType(clue_hits[maxEnergyIndex].getType());
    }
    clustersLayer.clear();
  }

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

  // Output CLUE clusters
  edm4hep::ClusterCollection* finalClusters = clustersHandle.createAndPut();

  // Output CLUE calo hits
  clue::CLUECalorimeterHitCollection clue_hit_coll_barrel;
  clue::CLUECalorimeterHitCollection clue_hit_coll_endcap;

  // Fill CLUECaloHits in the barrel
  if( EB_calo_coll->isValid() ) {
    for(const auto& calo_hit : (*EB_calo_coll) ){
      clue_hit_coll_barrel.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::barrel, bf.get( calo_hit.getCellID(), "layer")));
    }
  } else {
    throw std::runtime_error("Collection not found.");
  }
  debug() << EB_calo_coll->size() << " caloHits in " << EBCaloCollectionName << "." << endmsg;

  // Run CLUE in the barrel
  if(!clue_hit_coll_barrel.vect.empty()){

    std::map<int, std::vector<int> > clueClustersBarrel = runAlgo(clue_hit_coll_barrel.vect, true);
    debug() << "Produced " << clueClustersBarrel.size() << " clusters in " << EBCaloCollectionName << endmsg;
  
    clue_hit_coll.vect.insert(clue_hit_coll.vect.end(), clue_hit_coll_barrel.vect.begin(), clue_hit_coll_barrel.vect.end());

    fillFinalClusters(clue_hit_coll_barrel.vect, clueClustersBarrel, finalClusters);
    debug() << "Saved " << finalClusters->size() << " clusters using " << EBCaloCollectionName << endmsg;

  }

  // Total amount of EE+ and EE- layers (80)
  // already described in `include/CLDEndcapLayerTilesConstants.h` 
  int maxLayerPerSide = 40;

  // Fill CLUECaloHits in the endcap
  if( EE_calo_coll->isValid() ) {
    for(const auto& calo_hit : (*EE_calo_coll) ){
      if(bf.get( calo_hit.getCellID(), "side") < 0 || bf.get( calo_hit.getCellID(), "side") > 1){
        clue_hit_coll_endcap.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::endcap, bf.get( calo_hit.getCellID(), "layer")));
      } else { 
        clue_hit_coll_endcap.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::endcap, bf.get( calo_hit.getCellID(), "layer") + maxLayerPerSide));
      }
    }
  } else {
    throw std::runtime_error("Collection not found.");
  }
  debug() << EE_calo_coll->size() << " caloHits in " << EECaloCollectionName << "." << endmsg;

  // Run CLUE in the endcap
  if(!clue_hit_coll_endcap.vect.empty()){
    std::map<int, std::vector<int> > clueClustersEndcap = runAlgo(clue_hit_coll_endcap.vect, false);
    debug() << "Produced " << clueClustersEndcap.size() << " clusters in " << EECaloCollectionName << endmsg;
  
    clue_hit_coll.vect.insert(clue_hit_coll.vect.end(), clue_hit_coll_endcap.vect.begin(), clue_hit_coll_endcap.vect.end());

    fillFinalClusters(clue_hit_coll_endcap.vect, clueClustersEndcap, finalClusters);
    debug() << "Saved " << finalClusters->size() << " clusters using " << EECaloCollectionName << endmsg;

  }

  // Add cellID to CLUE clusters
  auto& clusters_md = m_podioDataSvc->getProvider().getCollectionMetaData(finalClusters->getID());
  clusters_md.setValue("CellIDEncodingString", cellIDstr);
  info() << "Saved " << finalClusters->size() << " CLUE clusters in total." << endmsg;

  // Save CLUE calo hits
  auto pCHV = std::make_unique<clue::CLUECalorimeterHitCollection>(clue_hit_coll);
  const StatusCode scStatusV = eventSvc()->registerObject("/Event/CLUECalorimeterHitCollection", pCHV.release());
  info() << "Saved " << clue_hit_coll.vect.size() << " CLUE calo hits in total. " << endmsg;

  // Save clusters as calo hits and add cellID to them
  edm4hep::CalorimeterHitCollection* finalCaloHits = caloHitsHandle.createAndPut();
  transformClustersInCaloHits(finalClusters, finalCaloHits);
  auto& calohits_md = m_podioDataSvc->getProvider().getCollectionMetaData(finalCaloHits->getID());
  calohits_md.setValue("CellIDEncodingString", cellIDstr);
  info() << "Saved " << finalCaloHits->size() << " clusters as calo hits" << endmsg;

  // Cleaning
  clue_hit_coll.vect.clear();
  cleanCLUEPoints();
 
  return StatusCode::SUCCESS;
}

StatusCode ClueGaudiAlgorithmWrapper::finalize() {
  return Algorithm::finalize();
}

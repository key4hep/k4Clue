#include "CLUEHistograms.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

DECLARE_COMPONENT(CLUEHistograms)

CLUEHistograms::CLUEHistograms(const std::string& name, ISvcLocator* svcLoc) : GaudiAlgorithm(name, svcLoc), m_eventDataSvc("EventDataSvc", "CLUEHistograms") {
  declareProperty("SaveEachEvent", saveEachEvent, "Allow to save further plots per event");
  declareProperty("ClusterCollection", ClusterCollectionName, "Collection of clusters in input");
  StatusCode sc = m_eventDataSvc.retrieve();
}

StatusCode CLUEHistograms::initialize() {
  info() << "CLUEHistograms::initialize()" << endmsg;
  if (GaudiAlgorithm::initialize().isFailure()) return StatusCode::FAILURE;

  m_podioDataSvc = dynamic_cast<PodioDataSvc*>(m_eventDataSvc.get());
  if (m_podioDataSvc == nullptr) {
    return StatusCode::FAILURE;
  }

  if (service("THistSvc", m_ths).isFailure()) {
    error() << "Couldn't get THistSvc" << endmsg;
    return StatusCode::FAILURE;
  }

  h_clusters = new TH1F("Clusters","Clusters",100, 0, 100);
  if (m_ths->regHist("/rec/Clusters", h_clusters).isFailure()) {
    error() << "Couldn't register Clusters hist" << endmsg;
  }

  h_clSize = new TH1F("ClustersSize","ClusterSize",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterSize", h_clSize).isFailure()) {
    error() << "Couldn't register ClusterSize hist" << endmsg;
  }

  h_clEnergy = new TH1F("ClusterEnergy","ClusterEnergy",100, 0, 0.100);
  if (m_ths->regHist("/rec/ClusterEnergy", h_clEnergy).isFailure()) {
    error() << "Couldn't register ClusterEnergy hist" << endmsg;
  }

  h_clLayer = new TH1F("ClusterLayer","ClusterLayer",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterLayer", h_clLayer).isFailure()) {
    error() << "Couldn't register ClusterLayer hist" << endmsg;
  }

  h_clHitsLayer = new TH1F("ClusterHitsLayer","ClusterHitsLayer",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterHitsLayer", h_clHitsLayer).isFailure()) {
    error() << "Couldn't register ClusterHitsLayer" << endmsg;
  }

  h_clHitsEnergyLayer = new TH1F("ClusterHitsEnergy_layer","ClusterHitsEnergy_layer",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterHitsEnergy_layer", h_clHitsEnergyLayer).isFailure()) {
    error() << "Couldn't register ClusterHitsEnergy_layer hist" << endmsg;
  }

  graphPosNames = {"Pos_clusters_XY", "Pos_clusters_YZ", "Pos_clusters_RZ",
                   "Pos_clusterHits_XY", "Pos_clusterHits_YZ", "Pos_clusterHits_RZ",
                   "Pos_followers_XY", "Pos_followers_YZ",
                   "Pos_seeds_XY", "Pos_seeds_YZ",
                   "Pos_outliers_XY", "Pos_outliers_YZ"};

  graphClueNames = {"Delta_Rho_followers", "Delta_Rho_seeds", "Delta_Rho_outliers" };

  for (auto iName : graphClueNames) {
    std::string nameGraphInFolder = "/rec/" + iName;
    TGraph* gr = new TGraph();
    graphClue.push_back(gr);
    gr->SetName(TString(iName));
    if (m_ths->regGraph(nameGraphInFolder, gr).isFailure()) {
      error() << "Couldn't register " << nameGraphInFolder << endmsg;
    }

  }

  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::execute() {
  info() << "CLUEHistograms::execute()" << endmsg;

  if(saveEachEvent){
    warning() << "Careful! You will be saving more plots than you actually want!" << endmsg;

    DataHandle<edm4hep::EventHeaderCollection> ev_handle {
      "EventHeader", Gaudi::DataHandle::Reader, this};
    auto evs = ev_handle.get();
    evNum = (*evs)[0].getEventNumber();
    info() << "Event number = " << evNum << endmsg;

    for (auto iName : graphPosNames) {
      std::string nameGraphInFolder = "/rec/Event" + std::to_string(evNum) + "/" + iName;
      TGraph* gr = new TGraph();
      graphPos[evNum].push_back(gr);
      gr->SetName(TString(iName));
      if (m_ths->regGraph(nameGraphInFolder, gr).isFailure()) {
        error() << "Couldn't register " << nameGraphInFolder << endmsg;
      }

    }

  } // endif saveEachEvent

  DataObject* pStatus  = nullptr;
  StatusCode  scStatus = eventSvc()->retrieveObject("/Event/CLUECalorimeterHitCollection", pStatus);
  if (scStatus.isSuccess()) {
    clue_calo_coll = static_cast<clue::CLUECalorimeterHitCollection*>(pStatus);
    info() << "CH SIZE : " << clue_calo_coll->vect.size() << endmsg;
  } else {
    info() << "Status NOT success" << endmsg;
  }

  // Read cluster collection
  DataHandle<edm4hep::ClusterCollection> cluster_handle {  
    ClusterCollectionName, Gaudi::DataHandle::Reader, this};
  cluster_coll = cluster_handle.get();

  // Get collection metadata cellID which is valid for both EB and EE
  auto collID = cluster_coll->getID();
  const auto cellIDstr = cluster_handle.getCollMetadataCellID(collID);
  const BitFieldCoder bf(cellIDstr);
  std::cout << cellIDstr << std::endl;

  std::uint64_t ch_layer = 0;
  std::uint64_t nClusters = 0;
  std::uint64_t nClusterHits = 0;

  h_clusters->Fill(cluster_coll->size());
  for (const auto& cl : *cluster_coll) {

    h_clEnergy->Fill(cl.getEnergy());
    h_clSize->Fill(cl.hits_size());
    if(saveEachEvent){
      double r = sqrt(cl.getPosition().x*cl.getPosition().x + cl.getPosition().y*cl.getPosition().y);
      graphPos[evNum][0]->SetPoint(nClusters, cl.getPosition().y, cl.getPosition().x);
      graphPos[evNum][1]->SetPoint(nClusters, cl.getPosition().z, cl.getPosition().y);
      graphPos[evNum][2]->SetPoint(nClusters, cl.getPosition().z, r);
    }

    for (const auto& hit : cl.getHits()) {
      ch_layer = bf.get( hit.getCellID(), "layer");
      h_clHitsLayer->Fill(ch_layer);
      h_clHitsEnergyLayer->Fill(ch_layer, hit.getEnergy());
      if(saveEachEvent){
        double r = sqrt(hit.getPosition().x*hit.getPosition().x + hit.getPosition().y*hit.getPosition().y);
        graphPos[evNum][3]->SetPoint(nClusterHits, hit.getPosition().y, hit.getPosition().x);
        graphPos[evNum][4]->SetPoint(nClusterHits, hit.getPosition().z, hit.getPosition().y);
        graphPos[evNum][5]->SetPoint(nClusterHits, hit.getPosition().z, r);
      }
      nClusterHits++;
    }


    h_clLayer->Fill(ch_layer);
    nClusters++;
  }

  std::uint64_t nSeeds = 0;
  std::uint64_t nFollowers = 0;
  std::uint64_t nOutliers = 0;
  for (const auto& clue_hit : (clue_calo_coll->vect)) {
    if(clue_hit.isFollower()){
      graphClue[0]->SetPoint(nFollowers_tot, clue_hit.getDelta(), clue_hit.getRho());
      if(saveEachEvent){
        graphPos[evNum][6]->SetPoint(nFollowers, clue_hit.getPosition().y, clue_hit.getPosition().x);
        graphPos[evNum][7]->SetPoint(nFollowers, clue_hit.getPosition().z, clue_hit.getPosition().y);
      }
      nFollowers++;
      nFollowers_tot++;
    }
    if(clue_hit.isSeed()){
      graphClue[1]->SetPoint(nSeeds_tot, clue_hit.getDelta(), clue_hit.getRho());
      if(saveEachEvent){
        graphPos[evNum][8]->SetPoint(nSeeds, clue_hit.getPosition().y, clue_hit.getPosition().x);
        graphPos[evNum][9]->SetPoint(nSeeds, clue_hit.getPosition().z, clue_hit.getPosition().y);
      }
      nSeeds++;
      nSeeds_tot++;
    }

    if(clue_hit.isOutlier()){
      graphClue[2]->SetPoint(nOutliers_tot, clue_hit.getDelta(), clue_hit.getRho());
      if(saveEachEvent){
        graphPos[evNum][10]->SetPoint(nOutliers, clue_hit.getPosition().y, clue_hit.getPosition().x);
        graphPos[evNum][11]->SetPoint(nOutliers, clue_hit.getPosition().z, clue_hit.getPosition().y);
      }
      nOutliers++;
      nOutliers_tot++;
    }
  }
  info() << nSeeds << " seeds." << endmsg;
  info() << nOutliers << " outliers." << endmsg;
  info() << nFollowers << " followers." << endmsg;
  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::finalize() {
  if (GaudiAlgorithm::finalize().isFailure()) return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

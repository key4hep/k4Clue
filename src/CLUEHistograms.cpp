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
                   "Pos_clusterHits_XY", "Pos_clusterHits_YZ", "Pos_clusterHits_RZ"
                   };

  t_hits = new TTree ("hits", "CLUE calo hits ntuple");
  if (m_ths->regTree("/rec/NtuplesHits", t_hits).isFailure()) {
    error() << "Couldn't register hits tree" << endmsg;
  }

  initializeTree();

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

  cleanTree();

  std::uint64_t nSeeds = 0;
  std::uint64_t nFollowers = 0;
  std::uint64_t nOutliers = 0;
  for (const auto& clue_hit : (clue_calo_coll->vect)) {
    m_event->push_back (evNum);
    if(clue_hit.inBarrel()){
      m_region->push_back (0);
    } else {
      m_region->push_back (1);
    }
    m_layer->push_back (clue_hit.getLayer());
    m_x->push_back (clue_hit.getPosition().x);
    m_y->push_back (clue_hit.getPosition().y);
    m_z->push_back (clue_hit.getPosition().z);
    m_eta->push_back (clue_hit.getEta());
    m_phi->push_back (clue_hit.getPhi());
    m_rho->push_back (clue_hit.getRho());
    m_delta->push_back (clue_hit.getDelta());

    if(clue_hit.isFollower()){
      m_status->push_back(1);
      nFollowers++;
    }
    if(clue_hit.isSeed()){
      m_status->push_back(2);
      nSeeds++;
    }

    if(clue_hit.isOutlier()){
      m_status->push_back(0);
      nOutliers++;
    }
  }
  info() << nSeeds << " seeds." << endmsg;
  info() << nOutliers << " outliers." << endmsg;
  info() << nFollowers << " followers." << endmsg;
  t_hits->Fill ();
  return StatusCode::SUCCESS;
}

void CLUEHistograms::initializeTree() {

  m_event = new std::vector<int>();
  m_region = new std::vector<int>();
  m_layer = new std::vector<int>();
  m_status = new std::vector<int>();
  m_x = new std::vector<float>();
  m_y = new std::vector<float>();
  m_z = new std::vector<float>();
  m_eta = new std::vector<float>();
  m_phi = new std::vector<float>();
  m_rho = new std::vector<float>();
  m_delta = new std::vector<float>();

  t_hits->Branch ("event", &m_event);
  t_hits->Branch ("region", &m_region);
  t_hits->Branch ("layer", &m_layer);
  t_hits->Branch ("status", &m_status);
  t_hits->Branch ("x", &m_x);
  t_hits->Branch ("y", &m_y);
  t_hits->Branch ("z", &m_z);
  t_hits->Branch ("eta", &m_eta);
  t_hits->Branch ("phi", &m_phi);
  t_hits->Branch ("rho", &m_rho);
  t_hits->Branch ("delta", &m_delta);

  return;
}

void CLUEHistograms::cleanTree() {
  m_event->clear();
  m_region->clear(); 
  m_layer->clear();
  m_status->clear();
  m_x->clear();
  m_y->clear();
  m_z->clear();
  m_eta->clear();
  m_phi->clear();
  m_rho->clear();
  m_delta ->clear();

  return;
}

StatusCode CLUEHistograms::finalize() {
  if (GaudiAlgorithm::finalize().isFailure()) return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

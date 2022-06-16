#include "CLUEHistograms.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

DECLARE_COMPONENT(CLUEHistograms)

CLUEHistograms::CLUEHistograms(const std::string& name, ISvcLocator* svcLoc) : GaudiAlgorithm(name, svcLoc), m_eventDataSvc("EventDataSvc", "CLUEHistograms") {
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

  t_hits = new TTree ("CLUEHits", "CLUE calo hits ntuple");
  if (m_ths->regTree("/rec/NtuplesHits", t_hits).isFailure()) {
    error() << "Couldn't register hits tree" << endmsg;
  }

  t_clusters = new TTree (TString(ClusterCollectionName), "Clusters ntuple");
  if (m_ths->regTree("/rec/"+ClusterCollectionName, t_clusters).isFailure()) {
    error() << "Couldn't register clusters tree" << endmsg;
  }

  std::string ClusterHitsCollectionName = ClusterCollectionName + "Hits";
  t_clhits = new TTree (TString(ClusterHitsCollectionName), "Clusters ntuple");
  if (m_ths->regTree("/rec/"+ClusterHitsCollectionName, t_clhits).isFailure()) {
    error() << "Couldn't register cluster hits tree" << endmsg;
  }

  initializeTrees();

  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::execute() {
  info() << "CLUEHistograms::execute()" << endmsg;

  DataHandle<edm4hep::EventHeaderCollection> ev_handle {
    "EventHeader", Gaudi::DataHandle::Reader, this};
  auto evs = ev_handle.get();
  evNum = (*evs)[0].getEventNumber();
  info() << "Event number = " << evNum << endmsg;

  DataObject* pStatus  = nullptr;
  StatusCode  scStatus = eventSvc()->retrieveObject("/Event/CLUECalorimeterHitCollection", pStatus);
  if (scStatus.isSuccess()) {
    clue_calo_coll = static_cast<clue::CLUECalorimeterHitCollection*>(pStatus);
    info() << "CH SIZE : " << clue_calo_coll->vect.size() << endmsg;
  } else {
    throw std::runtime_error("CLUE hits collection not available");
  }

  // Read EB collection for metadata cellID
  DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle {
    EBCaloCollectionName, Gaudi::DataHandle::Reader, this};
  EB_calo_coll = EB_calo_handle.get();

  // Read EE collection
  DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle {
    EECaloCollectionName, Gaudi::DataHandle::Reader, this};
  EE_calo_coll = EE_calo_handle.get();

  // Read cluster collection
  DataHandle<edm4hep::ClusterCollection> cluster_handle {  
    ClusterCollectionName, Gaudi::DataHandle::Reader, this};
  cluster_coll = cluster_handle.get();

  // Get collection metadata cellID which is valid for both EB, EE and Clusters
  auto collID = EB_calo_coll->getID();
  const auto cellIDstr = EB_calo_handle.getCollMetadataCellID(collID);
  const BitFieldCoder bf(cellIDstr);

  cleanTrees();

  std::uint64_t ch_layer = 0;
  std::uint64_t nClusters = 0;
  float totEnergy = 0;
  float totEnergyHits = 0;
  std::uint64_t totSize = 0;

  for (const auto& cl : *cluster_coll) {
    m_clusters_event->push_back (evNum);
    m_clusters_energy->push_back (cl.getEnergy());
    info() << "energy in cluster : " << cl.getEnergy() << endmsg; 
    m_clusters_size->push_back (cl.hits_size());

    m_clusters_x->push_back (cl.getPosition().x);
    m_clusters_y->push_back (cl.getPosition().y);
    m_clusters_z->push_back (cl.getPosition().z);

   for (const auto& hit : cl.getHits()) {
      ch_layer = bf.get( hit.getCellID(), "layer");
      m_clhits_event->push_back (evNum);
      m_clhits_layer->push_back (ch_layer);
      m_clhits_x->push_back (hit.getPosition().x);
      m_clhits_y->push_back (hit.getPosition().y);
      m_clhits_z->push_back (hit.getPosition().z);
      m_clhits_energy->push_back (hit.getEnergy());
      totEnergyHits += hit.getEnergy();

    }

    nClusters++;
    totEnergy += cl.getEnergy();
    totSize += cl.getHits().size();

  }
  m_clusters->push_back (nClusters);
  m_clusters_totEnergy->push_back (totEnergy);
  m_clusters_totEnergyHits->push_back (totEnergyHits);
  m_clusters_totSize->push_back (totSize);
  t_clusters->Fill ();
  t_clhits->Fill ();
  info() << ClusterCollectionName << " :: Total number hits = " << totSize << " with total energy (cl) = " << totEnergy << "; (hits) = " << totEnergyHits << endmsg; 

  std::uint64_t nSeeds = 0;
  std::uint64_t nFollowers = 0;
  std::uint64_t nOutliers = 0;
  totEnergy = 0;
  for (const auto& clue_hit : (clue_calo_coll->vect)) {
    m_hits_event->push_back (evNum);
    if(clue_hit.inBarrel()){
      m_hits_region->push_back (0);
    } else {
      m_hits_region->push_back (1);
    }
    m_hits_layer->push_back (clue_hit.getLayer());
    m_hits_x->push_back (clue_hit.getPosition().x);
    m_hits_y->push_back (clue_hit.getPosition().y);
    m_hits_z->push_back (clue_hit.getPosition().z);
    m_hits_eta->push_back (clue_hit.getEta());
    m_hits_phi->push_back (clue_hit.getPhi());
    m_hits_rho->push_back (clue_hit.getRho());
    m_hits_delta->push_back (clue_hit.getDelta());
    m_hits_energy->push_back (clue_hit.getEnergy());

    if(clue_hit.isFollower()){
      m_hits_status->push_back(1);
      totEnergy += clue_hit.getEnergy();
      nFollowers++;
    }
    if(clue_hit.isSeed()){
      m_hits_status->push_back(2);
      totEnergy += clue_hit.getEnergy();
      nSeeds++;
    }

    if(clue_hit.isOutlier()){
      m_hits_status->push_back(0);
      nOutliers++;
    }
  }
  info() << nSeeds << " seeds." << endmsg;
  info() << nOutliers << " outliers." << endmsg;
  info() << nFollowers << " followers." << endmsg;
  info() << totEnergy << " total energy." << endmsg;
  t_hits->Fill ();
  return StatusCode::SUCCESS;
}

void CLUEHistograms::initializeTrees() {

  m_hits_event = new std::vector<int>();
  m_hits_region = new std::vector<int>();
  m_hits_layer = new std::vector<int>();
  m_hits_status = new std::vector<int>();
  m_hits_x = new std::vector<float>();
  m_hits_y = new std::vector<float>();
  m_hits_z = new std::vector<float>();
  m_hits_eta = new std::vector<float>();
  m_hits_phi = new std::vector<float>();
  m_hits_rho = new std::vector<float>();
  m_hits_delta = new std::vector<float>();
  m_hits_energy = new std::vector<float>();

  t_hits->Branch ("event", &m_hits_event);
  t_hits->Branch ("region", &m_hits_region);
  t_hits->Branch ("layer", &m_hits_layer);
  t_hits->Branch ("status", &m_hits_status);
  t_hits->Branch ("x", &m_hits_x);
  t_hits->Branch ("y", &m_hits_y);
  t_hits->Branch ("z", &m_hits_z);
  t_hits->Branch ("eta", &m_hits_eta);
  t_hits->Branch ("phi", &m_hits_phi);
  t_hits->Branch ("rho", &m_hits_rho);
  t_hits->Branch ("delta", &m_hits_delta);
  t_hits->Branch ("energy", &m_hits_energy);

  m_clusters       = new std::vector<int>();
  m_clusters_event = new std::vector<int>();
  m_clusters_layer = new std::vector<int>();
  m_clusters_size  = new std::vector<int>();
  m_clusters_totSize  = new std::vector<int>();
  m_clusters_x = new std::vector<float>();
  m_clusters_y = new std::vector<float>();
  m_clusters_z = new std::vector<float>();
  m_clusters_energy = new std::vector<float>();
  m_clusters_totEnergy = new std::vector<float>();
  m_clusters_totEnergyHits = new std::vector<float>();

  t_clusters->Branch ("clusters", &m_clusters);
  t_clusters->Branch ("event", &m_clusters_event);
  t_clusters->Branch ("layer", &m_clusters_layer);
  t_clusters->Branch ("size", &m_clusters_size);
  t_clusters->Branch ("totSize", &m_clusters_totSize);
  t_clusters->Branch ("x", &m_clusters_x);
  t_clusters->Branch ("y", &m_clusters_y);
  t_clusters->Branch ("z", &m_clusters_z);
  t_clusters->Branch ("energy", &m_clusters_energy);
  t_clusters->Branch ("totEnergy", &m_clusters_totEnergy);
  t_clusters->Branch ("totEnergyHits", &m_clusters_totEnergyHits);

  m_clhits_event = new std::vector<int>();
  m_clhits_layer = new std::vector<int>();
  m_clhits_x = new std::vector<float>();
  m_clhits_y = new std::vector<float>();
  m_clhits_z = new std::vector<float>();
  m_clhits_energy = new std::vector<float>();

  t_clhits->Branch ("event", &m_clhits_event);
  t_clhits->Branch ("layer", &m_clhits_layer);
  t_clhits->Branch ("x", &m_clhits_x);
  t_clhits->Branch ("y", &m_clhits_y);
  t_clhits->Branch ("z", &m_clhits_z);
  t_clhits->Branch ("energy", &m_clhits_energy);

  return;
}

void CLUEHistograms::cleanTrees() {
  m_hits_event->clear();
  m_hits_region->clear(); 
  m_hits_layer->clear();
  m_hits_status->clear();
  m_hits_x->clear();
  m_hits_y->clear();
  m_hits_z->clear();
  m_hits_eta->clear();
  m_hits_phi->clear();
  m_hits_rho->clear();
  m_hits_delta->clear();
  m_hits_energy->clear();

  m_clusters->clear();
  m_clusters_event->clear();
  m_clusters_layer->clear();
  m_clusters_size->clear();
  m_clusters_totSize->clear();
  m_clusters_x->clear();
  m_clusters_y->clear();
  m_clusters_z->clear();
  m_clusters_energy->clear();
  m_clusters_totEnergy->clear();
  m_clusters_totEnergyHits->clear();

  m_clhits_event->clear();
  m_clhits_layer->clear();
  m_clhits_x->clear();
  m_clhits_y->clear();
  m_clhits_z->clear();
  m_clhits_energy->clear();

  return;
}

StatusCode CLUEHistograms::finalize() {
  if (GaudiAlgorithm::finalize().isFailure()) return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

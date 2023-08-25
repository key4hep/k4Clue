/*
 * Copyright (c) 2020-2023 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "CLUENtuplizer.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

DECLARE_COMPONENT(CLUENtuplizer)

CLUENtuplizer::CLUENtuplizer(const std::string& name, ISvcLocator* svcLoc) : GaudiAlgorithm(name, svcLoc), m_eventDataSvc("EventDataSvc", "CLUENtuplizer") {
  declareProperty("ClusterCollection", ClusterCollectionName, "Collection of clusters in input");
  declareProperty("BarrelCaloHitsCollection", EBCaloCollectionName, "Collection for Barrel Calo Hits used in input");
  declareProperty("EndcapCaloHitsCollection", EECaloCollectionName, "Collection for Endcap Calo Hits used in input");
  declareProperty("SingleMCParticle", singleMCParticle, "If this is True, the analysis is run only if one MCParticle is present in the event");
  StatusCode sc = m_eventDataSvc.retrieve();
}

StatusCode CLUENtuplizer::initialize() {
  if (GaudiAlgorithm::initialize().isFailure()) return StatusCode::FAILURE;

  m_podioDataSvc = dynamic_cast<PodioLegacyDataSvc*>(m_eventDataSvc.get());
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
    return StatusCode::FAILURE;
  }

  t_clusters = new TTree (TString(ClusterCollectionName), "Clusters ntuple");
  if (m_ths->regTree("/rec/"+ClusterCollectionName, t_clusters).isFailure()) {
    error() << "Couldn't register clusters tree" << endmsg;
    return StatusCode::FAILURE;
  }

  std::string ClusterHitsCollectionName = ClusterCollectionName + "Hits";
  t_clhits = new TTree (TString(ClusterHitsCollectionName), "Clusters ntuple");
  if (m_ths->regTree("/rec/"+ClusterHitsCollectionName, t_clhits).isFailure()) {
    error() << "Couldn't register cluster hits tree" << endmsg;
    return StatusCode::FAILURE;
  }

  initializeTrees();

  return StatusCode::SUCCESS;
}

StatusCode CLUENtuplizer::execute() {

  DataHandle<edm4hep::EventHeaderCollection> ev_handle {
    "EventHeader", Gaudi::DataHandle::Reader, this};
  auto evs = ev_handle.get();
  evNum = (*evs)[0].getEventNumber();
  info() << "Event number = " << evNum << std::endl;

  DataHandle<edm4hep::MCParticleCollection> mcp_handle {
    "MCParticles", Gaudi::DataHandle::Reader, this};
  auto mcps = mcp_handle.get();
  int mcps_primary = 0;
  float mcp_primary_energy = 0.f;
  std::for_each((*mcps).begin(), (*mcps).end(),
                [&mcps_primary, &mcp_primary_energy] (edm4hep::MCParticle mcp) { 
                  if(mcp.getGeneratorStatus() == 1){
                    mcps_primary += 1;
                    mcp_primary_energy = mcp.getEnergy();
                  }
                });
  info() << "MC Particles = " << mcps->size() << " (of which primaries = " << mcps_primary << ")" << endmsg;
  // If there is more than one primary, skip event
  if(singleMCParticle && mcps_primary > 1){
    warning() << "This event is skipped because there are " << mcps_primary << " primary MC particles." << endmsg;
    return StatusCode::SUCCESS;
  }

  DataObject* pStatus  = nullptr;
  StatusCode  scStatus = eventSvc()->retrieveObject("/Event/CLUECalorimeterHitCollection", pStatus);
  if (scStatus.isSuccess()) {
    clue_calo_coll = static_cast<clue::CLUECalorimeterHitCollection*>(pStatus);
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

  debug() << "ECAL Calorimeter Hits Size = " << int( EB_calo_coll->size()+EE_calo_coll->size() ) << endmsg;

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
  bool foundInECAL = false;

  for (const auto& cl : *cluster_coll) {
    m_clusters_event->push_back (evNum);
    m_clusters_energy->push_back (cl.getEnergy());
    m_clusters_size->push_back (cl.hits_size());

    m_clusters_x->push_back (cl.getPosition().x);
    m_clusters_y->push_back (cl.getPosition().y);
    m_clusters_z->push_back (cl.getPosition().z);

    // Sum up energy of cluster hits and save info
    // Printout the hits that are in Ecal but not included in the clusters
    int maxLayer = 0;
    for (const auto& hit : cl.getHits()) {
      foundInECAL = false;
      for (const auto& clEB : *EB_calo_coll) {
        if( clEB.getCellID() == hit.getCellID()){
          foundInECAL = true;
          break;  // Found in EB, break the loop
        }
        if(foundInECAL) {
          // Found in EB, break the loop
          break;
        } 
      }

      if(!foundInECAL){
        for (const auto& clEE : *EE_calo_coll) {
          if( clEE.getCellID() == hit.getCellID()){
            foundInECAL = true;
            break;  // Found in EE, break the loop
          }
          if(foundInECAL) {
            // Found in EE, break the loop
            break;
          }
        }
      }
      if(foundInECAL){
        ch_layer = bf.get( hit.getCellID(), "layer");
        maxLayer = std::max(int(ch_layer), maxLayer);
        //info() << "  ch cellID : " << hit.getCellID()
        //       << ", layer : " << ch_layer   
        //       << ", energy : " << hit.getEnergy() << endmsg; 
        m_clhits_event->push_back (evNum);
        m_clhits_layer->push_back (ch_layer);
        m_clhits_x->push_back (hit.getPosition().x);
        m_clhits_y->push_back (hit.getPosition().y);
        m_clhits_z->push_back (hit.getPosition().z);
        m_clhits_energy->push_back (hit.getEnergy());
        totEnergyHits += hit.getEnergy();
        totSize += 1;
      } else {
        debug() << "  This calo hit was NOT found among ECAL hits (cellID : " << hit.getCellID()
               << ", layer : " << ch_layer   
               << ", energy : " << hit.getEnergy() << " )" << endmsg; 
      }
    }
    nClusters++;
    if(!std::isnan(cl.getEnergy())){
      totEnergy += cl.getEnergy();
    }
    m_clusters_maxLayer->push_back (maxLayer);

  }
  m_clusters->push_back (nClusters);
  m_clusters_totEnergy->push_back (totEnergy);
  m_clusters_totEnergyHits->push_back (totEnergyHits);
  m_clusters_MCEnergy->push_back (mcp_primary_energy);
  m_clusters_totSize->push_back (totSize);
  t_clusters->Fill ();
  t_clhits->Fill ();
  info() << ClusterCollectionName << " : Total number hits = " << totSize << " with total energy (cl) = " << totEnergy << "; (hits) = " << totEnergyHits << endmsg; 

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
    m_hits_MCEnergy->push_back (mcp_primary_energy);

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
  debug() << "CLUE Calorimeter Hits Size = " << clue_calo_coll->vect.size() << endmsg;
  debug() << "Found: " << nSeeds << " seeds, "
         << nOutliers << " outliers, "
         << nFollowers << " followers." 
         << " Total energy clusterized: " << totEnergy << " GeV" << endmsg;
  t_hits->Fill ();
  return StatusCode::SUCCESS;
}

void CLUENtuplizer::initializeTrees() {

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
  m_hits_MCEnergy = new std::vector<float>();

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
  t_hits->Branch ("MCEnergy", &m_hits_MCEnergy);

  m_clusters          = new std::vector<int>();
  m_clusters_event    = new std::vector<int>();
  m_clusters_maxLayer = new std::vector<int>();
  m_clusters_size     = new std::vector<int>();
  m_clusters_totSize  = new std::vector<int>();
  m_clusters_x = new std::vector<float>();
  m_clusters_y = new std::vector<float>();
  m_clusters_z = new std::vector<float>();
  m_clusters_energy = new std::vector<float>();
  m_clusters_totEnergy = new std::vector<float>();
  m_clusters_totEnergyHits = new std::vector<float>();
  m_clusters_MCEnergy = new std::vector<float>();

  t_clusters->Branch ("clusters", &m_clusters);
  t_clusters->Branch ("event", &m_clusters_event);
  t_clusters->Branch ("maxLayer", &m_clusters_maxLayer);
  t_clusters->Branch ("size", &m_clusters_size);
  t_clusters->Branch ("totSize", &m_clusters_totSize);
  t_clusters->Branch ("x", &m_clusters_x);
  t_clusters->Branch ("y", &m_clusters_y);
  t_clusters->Branch ("z", &m_clusters_z);
  t_clusters->Branch ("energy", &m_clusters_energy);
  t_clusters->Branch ("totEnergy", &m_clusters_totEnergy);
  t_clusters->Branch ("totEnergyHits", &m_clusters_totEnergyHits);
  t_clusters->Branch ("MCEnergy", &m_clusters_MCEnergy);

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

void CLUENtuplizer::cleanTrees() {
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
  m_hits_MCEnergy->clear();

  m_clusters->clear();
  m_clusters_event->clear();
  m_clusters_maxLayer->clear();
  m_clusters_size->clear();
  m_clusters_totSize->clear();
  m_clusters_x->clear();
  m_clusters_y->clear();
  m_clusters_z->clear();
  m_clusters_energy->clear();
  m_clusters_totEnergy->clear();
  m_clusters_totEnergyHits->clear();
  m_clusters_MCEnergy->clear();

  m_clhits_event->clear();
  m_clhits_layer->clear();
  m_clhits_x->clear();
  m_clhits_y->clear();
  m_clhits_z->clear();
  m_clhits_energy->clear();

  return;
}

StatusCode CLUENtuplizer::finalize() {
  if (GaudiAlgorithm::finalize().isFailure()) return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

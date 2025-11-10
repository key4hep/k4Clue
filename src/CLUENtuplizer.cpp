/*
 * Copyright (c) 2020-2024 Key4hep-Project.
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
#include "k4FWCore/MetadataUtils.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep;
using namespace DDSegmentation;

DECLARE_COMPONENT(CLUENtuplizer)

StatusCode CLUENtuplizer::initialize() {
  const std::string ClusterCollectionName = inputLocations("InputClusters")[0];

  if (Gaudi::Algorithm::initialize().isFailure())
    return StatusCode::FAILURE;

  m_ths = service("THistSvc", true);
  if (!m_ths) {
    error() << "Couldn't get THistSvc" << endmsg;
    return StatusCode::FAILURE;
  }

  t_hits = new TTree("CLUEHits", "CLUE calo hits ntuple");
  if (m_ths->regTree("/rec/NtuplesHits", t_hits).isFailure()) {
    error() << "Couldn't register hits tree" << endmsg;
    return StatusCode::FAILURE;
  }

  t_clusters = new TTree(TString(ClusterCollectionName), "Clusters ntuple");
  if (m_ths->regTree("/rec/" + ClusterCollectionName, t_clusters).isFailure()) {
    error() << "Couldn't register clusters tree" << endmsg;
    return StatusCode::FAILURE;
  }

  std::string ClusterHitsCollectionName = ClusterCollectionName + "Hits";
  t_clhits = new TTree(TString(ClusterHitsCollectionName), "Clusters ntuple");
  if (m_ths->regTree("/rec/" + ClusterHitsCollectionName, t_clhits).isFailure()) {
    error() << "Couldn't register cluster hits tree" << endmsg;
    return StatusCode::FAILURE;
  }

  t_MCParticles = new TTree("MCParticles", "Monte Carlo Particles ntuple");
  if (m_ths->regTree("/rec/MCParticles", t_MCParticles).isFailure()) {
    error() << "Couldn't register MC Particles tree" << endmsg;
    return StatusCode::FAILURE;
  }

  t_links = new TTree("associations", "associations ntuple");
  if (m_ths->regTree("/rec/associations", t_links).isFailure()) {
    error() << "Couldn't register associations tree" << endmsg;
    return StatusCode::FAILURE;
  }

  initializeTrees();

  return StatusCode::SUCCESS;
}

void CLUENtuplizer::operator()(const CaloHitColl& EB_calo_coll, const CaloHitColl& EE_calo_coll,
                               const ClusterColl& cluster_coll,
                               const edm4hep::EventHeaderCollection& evs, const MCPartColl& mcps,
                               const ClusterMCLinkColl& linksClus) const {
  const std::string ClusterCollectionName = inputLocations("InputClusters")[0];
  evNum = evs[0].getEventNumber();
  info() << "Event number = " << evNum << endmsg;

  int mcps_primary = 0;
  std::for_each(mcps.begin(), mcps.end(), [&mcps_primary](edm4hep::MCParticle mcp) {
    if (mcp.getGeneratorStatus() == 1) {
      mcps_primary += 1;
    }
  });
  info() << "MC Particles = " << mcps.size() << " (of which primaries = " << mcps_primary << ")" << endmsg;

  DataObject* pStatus = nullptr;
  StatusCode scStatus = eventSvc()->retrieveObject("/Event/CLUECalorimeterHitCollection", pStatus);
  clue::CLUECalorimeterHitCollection* clue_calo_coll;
  if (scStatus.isSuccess()) {
    clue_calo_coll = static_cast<clue::CLUECalorimeterHitCollection*>(pStatus);
  } else {
    throw std::runtime_error("CLUE hits collection not available");
  }

  debug() << "ECAL Calorimeter Hits Size = " << EB_calo_coll.size() + EE_calo_coll.size() << endmsg;

  // Get collection metadata cellID which is valid for both EB, EE and Clusters
  const std::string cellIDstr = k4FWCore::getParameter<std::string>(podio::collMetadataParamName("ECalBarrelCollection", edm4hep::labels::CellIDEncoding), this).value_or("");
  const BitFieldCoder bf(cellIDstr);
  cleanTrees();

  // use the already built cluster to MC particle associators
  std::multimap<uint32_t, std::pair<uint32_t, float>> simToRecoLink;
  std::multimap<uint32_t, std::pair<uint32_t, float>> recoToSimLink;
  for (const auto& link : linksClus) {
    const auto recIdx = link.getFrom().getObjectID().index;
    const auto simIdx = link.getTo().getObjectID().index;
    const auto weight = link.getWeight();
    simToRecoLink.emplace(simIdx, std::make_pair(recIdx, weight));
    recoToSimLink.emplace(recIdx, std::make_pair(simIdx, weight));
  }

  std::vector<int> simIdMapping(mcps.size(), -1);
  int id = 0;
  for (std::size_t simId = 0; simId < mcps.size(); ++simId) {
    if (simToRecoLink.contains(simId)) {
      simIdMapping[simId] = id;
      ++id;
      const auto& mcp = mcps[simId];
      m_sim_event.push_back(evNum);
      m_sim_pdg.push_back(mcp.getPDG());
      m_sim_charge.push_back(mcp.getCharge());
      m_sim_vtx_x.push_back(mcp.getVertex().x);
      m_sim_vtx_y.push_back(mcp.getVertex().y);
      m_sim_vtx_z.push_back(mcp.getVertex().z);
      m_sim_momentum_x.push_back(mcp.getMomentum().x);
      m_sim_momentum_y.push_back(mcp.getMomentum().y);
      m_sim_momentum_z.push_back(mcp.getMomentum().z);
      m_sim_time.push_back(mcp.getTime());
      m_sim_energy.push_back(mcp.getEnergy());
      m_sim_primary.push_back(mcp.getGeneratorStatus() == 1);
      std::vector<int> ids;
      std::vector<float> shEn;
      const auto size = simToRecoLink.count(simId);
      ids.reserve(size);
      shEn.reserve(size);
      auto range = simToRecoLink.equal_range(simId);
      std::for_each(range.first, range.second, [&ids, &shEn](const auto& pair) {
        ids.push_back(pair.second.first);
        shEn.push_back(pair.second.second);
      });
      m_simToReco_index.push_back(ids);
      m_simToReco_sharedEnergy.push_back(shEn);
    }
  }

  for (std::size_t recoId = 0; recoId < cluster_coll.size(); ++recoId) {
    std::vector<int> ids;
    std::vector<float> shEn;
    const auto size = recoToSimLink.count(recoId);
    ids.reserve(size);
    shEn.reserve(size);
    auto range = recoToSimLink.equal_range(recoId);
    std::for_each(range.first, range.second, [&](const auto& pair) {
      if (simIdMapping[pair.second.first] == -1)
        throw error() << "No SimToReco but RecoToSim for simparticle " << pair.first << endmsg;
      ids.push_back(simIdMapping[pair.second.first]);
      shEn.push_back(pair.second.second);
    });
    m_recoToSim_index.push_back(ids);
    m_recoToSim_sharedEnergy.push_back(shEn);
  }

  t_MCParticles->Fill();
  t_links->Fill();

  std::uint64_t ch_layer = 0;
  std::uint64_t nClusters = 0;
  float totEnergy = 0;
  float totEnergyHits = 0;
  std::uint64_t totSize = 0;

  info() << ClusterCollectionName << " : Total number of clusters =  " << int(cluster_coll.size()) << endmsg;
  for (const auto& cl : cluster_coll) {
    m_clusters_event.push_back(evNum);
    m_clusters_energy.push_back(cl.getEnergy());
    m_clusters_size.push_back(cl.hits_size());

    m_clusters_x.push_back(cl.getPosition().x);
    m_clusters_y.push_back(cl.getPosition().y);
    m_clusters_z.push_back(cl.getPosition().z);

    // Sum up energy of cluster hits and save info
    int maxLayer = 0;
    std::vector<float> hits_time;
    for (const auto& hit : cl.getHits()) {
      ch_layer = bf.get(hit.getCellID(), "layer");
      maxLayer = std::max(int(ch_layer), maxLayer);
      verbose() << "  ch cellID : " << hit.getCellID() << ", layer : " << ch_layer << ", energy : " << hit.getEnergy()
                << endmsg;
      m_clhits_event.push_back(evNum);
      m_clhits_layer.push_back(ch_layer);
      m_clhits_x.push_back(hit.getPosition().x);
      m_clhits_y.push_back(hit.getPosition().y);
      m_clhits_z.push_back(hit.getPosition().z);
      m_clhits_time.push_back(hit.getTime());
      hits_time.push_back(hit.getTime());
      m_clhits_energy.push_back(hit.getEnergy());
      m_clhits_id.push_back(nClusters);
      totEnergyHits += hit.getEnergy();
      totSize += 1;
    }
    nClusters++;
    if (!std::isnan(cl.getEnergy())) {
      totEnergy += cl.getEnergy();
    }
    m_clusters_maxLayer.push_back(maxLayer);
    m_clusters_time.push_back(std::accumulate(hits_time.begin(), hits_time.end(), 0.f) / hits_time.size());
  }
  m_clusters.push_back(nClusters);
  m_clusters_totEnergy.push_back(totEnergy);
  m_clusters_totEnergyHits.push_back(totEnergyHits);
  m_clusters_totSize.push_back(totSize);
  t_clusters->Fill();
  t_clhits->Fill();
  info() << ClusterCollectionName << " : Total number hits = " << totSize << " with total energy (cl) = " << totEnergy
         << "; (hits) = " << totEnergyHits << endmsg;

  const auto& clue_calo_coll_vect = clue_calo_coll->vect;

  std::uint64_t nSeeds = 0;
  std::uint64_t nFollowers = 0;
  std::uint64_t nOutliers = 0;
  totEnergy = 0;
  debug() << "CLUE Calorimeter Hits Size = " << clue_calo_coll_vect.size() << endmsg;
  for (const auto& clue_hit : (clue_calo_coll_vect)) {
    m_hits_event.push_back(evNum);
    if (clue_hit.inBarrel()) {
      m_hits_region.push_back(0);
    } else {
      m_hits_region.push_back(1);
    }
    m_hits_clusId.push_back(clue_hit.getClusterIndex());
    m_hits_layer.push_back(clue_hit.getLayer());
    m_hits_x.push_back(clue_hit.getPosition().x);
    m_hits_y.push_back(clue_hit.getPosition().y);
    m_hits_z.push_back(clue_hit.getPosition().z);
    m_hits_eta.push_back(clue_hit.getEta());
    m_hits_phi.push_back(clue_hit.getPhi());
    m_hits_rho.push_back(clue_hit.getRho());
    m_hits_delta.push_back(clue_hit.getDelta());
    m_hits_time.push_back(clue_hit.getTime());
    m_hits_energy.push_back(clue_hit.getEnergy());

    if (clue_hit.isFollower()) {
      m_hits_status.push_back(1);
      totEnergy += clue_hit.getEnergy();
      nFollowers++;
    }
    if (clue_hit.isSeed()) {
      m_hits_status.push_back(2);
      totEnergy += clue_hit.getEnergy();
      nSeeds++;
    }

    if (clue_hit.isOutlier()) {
      m_hits_status.push_back(0);
      nOutliers++;
    }
  }
  debug() << "Found: " << nSeeds << " seeds, " << nOutliers << " outliers, " << nFollowers << " followers."
          << " Total energy clusterized: " << totEnergy << " GeV" << endmsg;
  t_hits->Fill();
}

void CLUENtuplizer::initializeTrees() {
  t_hits->Branch("event", &m_hits_event);
  t_hits->Branch("region", &m_hits_region);
  t_hits->Branch("layer", &m_hits_layer);
  t_hits->Branch("status", &m_hits_status);
  t_hits->Branch("clusterId", &m_hits_clusId);
  t_hits->Branch("x", &m_hits_x);
  t_hits->Branch("y", &m_hits_y);
  t_hits->Branch("z", &m_hits_z);
  t_hits->Branch("eta", &m_hits_eta);
  t_hits->Branch("phi", &m_hits_phi);
  t_hits->Branch("rho", &m_hits_rho);
  t_hits->Branch("delta", &m_hits_delta);
  t_hits->Branch("time", &m_hits_time);
  t_hits->Branch("energy", &m_hits_energy);

  t_clusters->Branch("clusters", &m_clusters);
  t_clusters->Branch("event", &m_clusters_event);
  t_clusters->Branch("maxLayer", &m_clusters_maxLayer);
  t_clusters->Branch("size", &m_clusters_size);
  t_clusters->Branch("totSize", &m_clusters_totSize);
  t_clusters->Branch("x", &m_clusters_x);
  t_clusters->Branch("y", &m_clusters_y);
  t_clusters->Branch("z", &m_clusters_z);
  t_clusters->Branch("time", &m_clusters_time);
  t_clusters->Branch("energy", &m_clusters_energy);
  t_clusters->Branch("totEnergy", &m_clusters_totEnergy);
  t_clusters->Branch("totEnergyHits", &m_clusters_totEnergyHits);

  t_clhits->Branch("event", &m_clhits_event);
  t_clhits->Branch("layer", &m_clhits_layer);
  t_clhits->Branch("x", &m_clhits_x);
  t_clhits->Branch("y", &m_clhits_y);
  t_clhits->Branch("z", &m_clhits_z);
  t_clhits->Branch("time", &m_clhits_time);
  t_clhits->Branch("energy", &m_clhits_energy);
  t_clhits->Branch("clusterId", &m_clhits_id);

  t_MCParticles->Branch("event", &m_sim_event);
  t_MCParticles->Branch("pdg", &m_sim_pdg);
  t_MCParticles->Branch("charge", &m_sim_charge);
  t_MCParticles->Branch("vertex_x", &m_sim_vtx_x);
  t_MCParticles->Branch("vertex_y", &m_sim_vtx_y);
  t_MCParticles->Branch("vertex_z", &m_sim_vtx_z);
  t_MCParticles->Branch("p_x", &m_sim_momentum_x);
  t_MCParticles->Branch("p_y", &m_sim_momentum_y);
  t_MCParticles->Branch("p_z", &m_sim_momentum_z);
  t_MCParticles->Branch("time", &m_sim_time);
  t_MCParticles->Branch("energy", &m_sim_energy);
  t_MCParticles->Branch("primary", &m_sim_primary);

  t_links->Branch("simToRecoIndex", &m_simToReco_index);
  t_links->Branch("simToRecoEnergy", &m_simToReco_sharedEnergy);
  t_links->Branch("recoToSimIndex", &m_recoToSim_index);
  t_links->Branch("recoToSimEnergy", &m_recoToSim_sharedEnergy);

  return;
}

void CLUENtuplizer::cleanTrees() const {
  m_hits_event.clear();
  m_hits_region.clear();
  m_hits_layer.clear();
  m_hits_status.clear();
  m_hits_clusId.clear();
  m_hits_x.clear();
  m_hits_y.clear();
  m_hits_z.clear();
  m_hits_eta.clear();
  m_hits_phi.clear();
  m_hits_rho.clear();
  m_hits_delta.clear();
  m_hits_time.clear();
  m_hits_energy.clear();

  m_clusters.clear();
  m_clusters_event.clear();
  m_clusters_maxLayer.clear();
  m_clusters_size.clear();
  m_clusters_totSize.clear();
  m_clusters_x.clear();
  m_clusters_y.clear();
  m_clusters_z.clear();
  m_clusters_time.clear();
  m_clusters_energy.clear();
  m_clusters_totEnergy.clear();
  m_clusters_totEnergyHits.clear();

  m_clhits_event.clear();
  m_clhits_layer.clear();
  m_clhits_x.clear();
  m_clhits_y.clear();
  m_clhits_z.clear();
  m_clhits_time.clear();
  m_clhits_energy.clear();
  m_clhits_id.clear();

  m_sim_event.clear();
  m_sim_pdg.clear();
  m_sim_charge.clear();
  m_sim_vtx_x.clear();
  m_sim_vtx_y.clear();
  m_sim_vtx_z.clear();
  m_sim_momentum_x.clear();
  m_sim_momentum_y.clear();
  m_sim_momentum_z.clear();
  m_sim_time.clear();
  m_sim_energy.clear();
  m_sim_primary.clear();

  m_simToReco_index.clear();
  m_simToReco_sharedEnergy.clear();
  m_recoToSim_index.clear();
  m_recoToSim_sharedEnergy.clear();

  return;
}

StatusCode CLUENtuplizer::finalize() {
  if (Gaudi::Algorithm::finalize().isFailure())
    return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

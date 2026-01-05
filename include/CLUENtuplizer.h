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
#ifndef CLUE_HISTOGRAMS_H
#define CLUE_HISTOGRAMS_H

#include "Gaudi/Property.h"
#include "GaudiKernel/ITHistSvc.h"
#include "k4FWCore/Consumer.h"

#include "CLUECalorimeterHit.h"
#include <edm4hep/CaloHitSimCaloHitLinkCollection.h>
#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/ClusterMCParticleLinkCollection.h>
#include <edm4hep/Constants.h>
#include <edm4hep/EventHeaderCollection.h>
#include <edm4hep/MCParticleCollection.h>

#include "TH1F.h"
#include "TTree.h"

using CaloHitColl = edm4hep::CalorimeterHitCollection;
using ClueHitColl = clue::CLUECalorimeterHitCollection;
using ClusterColl = edm4hep::ClusterCollection;
using MCPartColl = edm4hep::MCParticleCollection;
using ClusterMCLinkColl = edm4hep::ClusterMCParticleLinkCollection;

struct CLUENtuplizer final
    : k4FWCore::Consumer<void(const CaloHitColl& EB_calo_coll, const CaloHitColl& EE_calo_coll,
                              const ClusterColl& cluster_coll, const edm4hep::EventHeaderCollection& ev_handle,
                              const MCPartColl& mcp_handle, const ClusterMCLinkColl& clustersLink_handle)> {
  CLUENtuplizer(const std::string& name, ISvcLocator* svcLoc)
      : Consumer(name, svcLoc,
                 {KeyValues("BarrelCaloHitsCollection", {"ECALBarrel"}),
                  KeyValues("EndcapCaloHitsCollection", {"ECALEndcap"}), KeyValues("InputClusters", {"CLUEClusters"}),
                  KeyValue("EventHeader", "EventHeader"), KeyValue("MCParticles", "MCParticles"),
                  KeyValues("ClusterLinks", {"ClusterMCTruthLink"})}) {}

  /// Initialize.
  StatusCode initialize() override;
  /// Initialize tree.
  void initializeTrees();
  /// Clean tree.
  void cleanTrees() const;
  /// Finalize.
  StatusCode finalize() override;

  void operator()(const CaloHitColl& EB_calo_coll, const CaloHitColl& EE_calo_coll, const ClusterColl& cluster_coll,
                  const edm4hep::EventHeaderCollection& evs, const MCPartColl& mcps,
                  const ClusterMCLinkColl& linksClus) const override;

private:
  Gaudi::Property<std::string> m_CLUECaloHitCollName{this, "CLUEHitCollName", "CLUECalorimeterHitCollection",
                                                     "Name of the collection of CLUE calorimeter hits"};
  SmartIF<ITHistSvc> m_ths; ///< THistogram service

  mutable TTree* t_hits{nullptr};
  mutable std::vector<int> m_hits_event;
  mutable std::vector<int> m_hits_region;
  mutable std::vector<int> m_hits_layer;
  mutable std::vector<int> m_hits_status;
  mutable std::vector<int> m_hits_clusId;
  mutable std::vector<float> m_hits_x;
  mutable std::vector<float> m_hits_y;
  mutable std::vector<float> m_hits_z;
  mutable std::vector<float> m_hits_eta;
  mutable std::vector<float> m_hits_phi;
  mutable std::vector<float> m_hits_rho;
  mutable std::vector<float> m_hits_delta;
  mutable std::vector<float> m_hits_time;
  mutable std::vector<float> m_hits_energy;

  mutable TTree* t_clusters{nullptr};
  mutable std::vector<int> m_clusters;
  mutable std::vector<int> m_clusters_event;
  mutable std::vector<int> m_clusters_maxLayer;
  mutable std::vector<int> m_clusters_size;
  mutable std::vector<int> m_clusters_totSize;
  mutable std::vector<float> m_clusters_x;
  mutable std::vector<float> m_clusters_y;
  mutable std::vector<float> m_clusters_z;
  mutable std::vector<float> m_clusters_time;
  mutable std::vector<float> m_clusters_energy;
  mutable std::vector<float> m_clusters_totEnergy;
  mutable std::vector<float> m_clusters_totEnergyHits;

  mutable TTree* t_clhits{nullptr};
  mutable std::vector<int> m_clhits_event;
  mutable std::vector<int> m_clhits_layer;
  mutable std::vector<int> m_clhits_id;
  mutable std::vector<float> m_clhits_x;
  mutable std::vector<float> m_clhits_y;
  mutable std::vector<float> m_clhits_z;
  mutable std::vector<float> m_clhits_time;
  mutable std::vector<float> m_clhits_energy;

  mutable TTree* t_MCParticles{nullptr};
  mutable std::vector<int> m_sim_event;
  mutable std::vector<int> m_sim_pdg;
  mutable std::vector<int> m_sim_charge;
  mutable std::vector<float> m_sim_vtx_x;
  mutable std::vector<float> m_sim_vtx_y;
  mutable std::vector<float> m_sim_vtx_z;
  mutable std::vector<float> m_sim_momentum_x;
  mutable std::vector<float> m_sim_momentum_y;
  mutable std::vector<float> m_sim_momentum_z;
  mutable std::vector<float> m_sim_time;
  mutable std::vector<float> m_sim_energy;
  mutable std::vector<bool> m_sim_primary;

  mutable TTree* t_links{nullptr};
  mutable std::vector<std::vector<int>> m_simToReco_index;
  mutable std::vector<std::vector<float>> m_simToReco_sharedEnergy;
  mutable std::vector<std::vector<int>> m_recoToSim_index;
  mutable std::vector<std::vector<float>> m_recoToSim_sharedEnergy;

  mutable std::int32_t evNum;
};

#endif // CLUE_HISTOGRAMS_H

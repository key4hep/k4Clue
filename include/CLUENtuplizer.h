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

#include "Gaudi/Algorithm.h"
#include "GaudiKernel/ITHistSvc.h"
#include "k4FWCore/DataHandle.h"
#include "k4FWCore/MetaDataHandle.h"

#include "CLUECalorimeterHit.h"
#include <edm4hep/CaloHitSimCaloHitLinkCollection.h>
#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/ClusterMCParticleLinkCollection.h>
#include <edm4hep/Constants.h>
#include <edm4hep/EventHeaderCollection.h>
#include <edm4hep/MCParticleCollection.h>

#include "TGraph.h"
#include "TH1F.h"

class CLUENtuplizer : public Gaudi::Algorithm {

public:
  /// Constructor.
  CLUENtuplizer(const std::string& name, ISvcLocator* svcLoc);
  /// Initialize.
  virtual StatusCode initialize();
  /// Initialize tree.
  void initializeTrees();
  /// Clean tree.
  void cleanTrees() const;
  /// Execute.
  virtual StatusCode execute(const EventContext&) const;
  /// Finalize.
  virtual StatusCode finalize();

private:
  mutable const clue::CLUECalorimeterHitCollection* clue_calo_coll;
  std::string ClusterCollectionName;
  mutable const edm4hep::ClusterCollection* cluster_coll;
  mutable const edm4hep::CalorimeterHitCollection* EB_calo_coll;
  mutable const edm4hep::CalorimeterHitCollection* EE_calo_coll;
  mutable k4FWCore::DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle{"BarrelInputHits",
                                                                                 Gaudi::DataHandle::Reader, this};
  mutable k4FWCore::DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle{"EndcapInputHits",
                                                                                 Gaudi::DataHandle::Reader, this};
  mutable k4FWCore::DataHandle<edm4hep::EventHeaderCollection> ev_handle{"EventHeader", Gaudi::DataHandle::Reader,
                                                                         this};
  mutable k4FWCore::DataHandle<edm4hep::MCParticleCollection> mcp_handle{"MCParticles", Gaudi::DataHandle::Reader,
                                                                         this};
  mutable k4FWCore::DataHandle<edm4hep::CaloHitSimCaloHitLinkCollection> relationHitLink_handle{
      "RelationCaloHit", Gaudi::DataHandle::Reader, this};
  mutable k4FWCore::DataHandle<edm4hep::ClusterMCParticleLinkCollection> clustersLink_handle{"ClusterMCTruthLink",
                                                                                     Gaudi::DataHandle::Reader, this};
  k4FWCore::MetaDataHandle<std::string> cellIDHandle{EB_calo_handle, edm4hep::labels::CellIDEncoding,
                                                     Gaudi::DataHandle::Reader};

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

  mutable TTree* t_links{nullptr};
  mutable std::vector<std::vector<int>> m_simToReco_index;
  mutable std::vector<std::vector<float>> m_simToReco_sharedEnergy;
  mutable std::vector<std::vector<int>> m_recoToSim_index;
  mutable std::vector<std::vector<float>> m_recoToSim_sharedEnergy;

  mutable std::int32_t evNum;
};

#endif // CLUE_HISTOGRAMS_H

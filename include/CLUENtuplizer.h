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
#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/Constants.h>
#include <edm4hep/EventHeaderCollection.h>
#include <edm4hep/MCParticleCollection.h>

#include "TGraph.h"
#include "TH1F.h"

class CLUENtuplizer : public Gaudi::Algorithm {

public:
  /// Constructor.
  CLUENtuplizer(const std::string& name, ISvcLocator* svcLoc);
  /// Destructor.
  ~CLUENtuplizer() {
    delete m_hits_event;
    delete m_hits_region;
    delete m_hits_layer;
    delete m_hits_status;
    delete m_hits_x;
    delete m_hits_y;
    delete m_hits_z;
    delete m_hits_eta;
    delete m_hits_phi;
    delete m_hits_rho;
    delete m_hits_delta;
    delete m_hits_energy;
    delete m_hits_MCEnergy;

    delete m_clusters;
    delete m_clusters_event;
    delete m_clusters_maxLayer;
    delete m_clusters_size;
    delete m_clusters_totSize;
    delete m_clusters_x;
    delete m_clusters_y;
    delete m_clusters_z;
    delete m_clusters_energy;
    delete m_clusters_totEnergy;
    delete m_clusters_totEnergyHits;
    delete m_clusters_MCEnergy;

    delete m_clhits_event;
    delete m_clhits_layer;
    delete m_clhits_x;
    delete m_clhits_y;
    delete m_clhits_z;
    delete m_clhits_energy;
  };
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
  mutable DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle{"BarrelInputHits", Gaudi::DataHandle::Reader,
                                                                       this};
  mutable DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle{"EndcapInputHits", Gaudi::DataHandle::Reader,
                                                                       this};
  mutable DataHandle<edm4hep::EventHeaderCollection> ev_handle{"EventHeader", Gaudi::DataHandle::Reader, this};
  mutable DataHandle<edm4hep::MCParticleCollection> mcp_handle{"MCParticles", Gaudi::DataHandle::Reader, this};
  MetaDataHandle<std::string> cellIDHandle{EB_calo_handle, edm4hep::labels::CellIDEncoding, Gaudi::DataHandle::Reader};

  bool singleMCParticle = false;

  SmartIF<ITHistSvc> m_ths; ///< THistogram service

  mutable TTree* t_hits{nullptr};
  mutable std::vector<int>* m_hits_event = nullptr;
  mutable std::vector<int>* m_hits_region = nullptr;
  mutable std::vector<int>* m_hits_layer = nullptr;
  mutable std::vector<int>* m_hits_status = nullptr;
  mutable std::vector<float>* m_hits_x = nullptr;
  mutable std::vector<float>* m_hits_y = nullptr;
  mutable std::vector<float>* m_hits_z = nullptr;
  mutable std::vector<float>* m_hits_eta = nullptr;
  mutable std::vector<float>* m_hits_phi = nullptr;
  mutable std::vector<float>* m_hits_rho = nullptr;
  mutable std::vector<float>* m_hits_delta = nullptr;
  mutable std::vector<float>* m_hits_energy = nullptr;
  mutable std::vector<float>* m_hits_MCEnergy = nullptr;

  mutable TTree* t_clusters{nullptr};
  mutable std::vector<int>* m_clusters = nullptr;
  mutable std::vector<int>* m_clusters_event = nullptr;
  mutable std::vector<int>* m_clusters_maxLayer = nullptr;
  mutable std::vector<int>* m_clusters_size = nullptr;
  mutable std::vector<int>* m_clusters_totSize = nullptr;
  mutable std::vector<float>* m_clusters_x = nullptr;
  mutable std::vector<float>* m_clusters_y = nullptr;
  mutable std::vector<float>* m_clusters_z = nullptr;
  mutable std::vector<float>* m_clusters_energy = nullptr;
  mutable std::vector<float>* m_clusters_totEnergy = nullptr;
  mutable std::vector<float>* m_clusters_totEnergyHits = nullptr;
  mutable std::vector<float>* m_clusters_MCEnergy = nullptr;

  mutable TTree* t_clhits{nullptr};
  mutable std::vector<int>* m_clhits_event = nullptr;
  mutable std::vector<int>* m_clhits_layer = nullptr;
  mutable std::vector<float>* m_clhits_x = nullptr;
  mutable std::vector<float>* m_clhits_y = nullptr;
  mutable std::vector<float>* m_clhits_z = nullptr;
  mutable std::vector<float>* m_clhits_energy = nullptr;

  mutable std::int32_t evNum;
};

#endif // CLUE_HISTOGRAMS_H

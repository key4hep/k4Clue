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
#ifndef CLUE_HISTOGRAMS_H
#define CLUE_HISTOGRAMS_H

#include "k4FWCore/DataHandle.h"
#include "GaudiAlg/GaudiAlgorithm.h"
#include "GaudiKernel/ITHistSvc.h"

#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/MCParticleCollection.h>
#include <edm4hep/EventHeaderCollection.h>
#include "CLUECalorimeterHit.h"

#include "TH1F.h"
#include "TGraph.h"

class CLUENtuplizer : public GaudiAlgorithm {

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
  void cleanTrees();
  /// Execute.
  virtual StatusCode execute();
  /// Finalize.
  virtual StatusCode finalize();


private:
  const clue::CLUECalorimeterHitCollection* clue_calo_coll;
  std::string ClusterCollectionName;
  const edm4hep::ClusterCollection* cluster_coll; 
  std::string EBCaloCollectionName = "ECALBarrel";
  std::string EECaloCollectionName = "ECALEndcap";
  const edm4hep::CalorimeterHitCollection* EB_calo_coll;
  const edm4hep::CalorimeterHitCollection* EE_calo_coll;

  bool singleMCParticle = false;

  // PODIO data service
  ServiceHandle<IDataProviderSvc> m_eventDataSvc;
  PodioLegacyDataSvc* m_podioDataSvc;

  ITHistSvc* m_ths{nullptr};  ///< THistogram service

  TTree* t_hits{nullptr};
  std::vector<int> *m_hits_event = nullptr;
  std::vector<int> *m_hits_region = nullptr;
  std::vector<int> *m_hits_layer = nullptr;
  std::vector<int> *m_hits_status = nullptr;
  std::vector<float> *m_hits_x = nullptr;
  std::vector<float> *m_hits_y = nullptr;
  std::vector<float> *m_hits_z = nullptr;
  std::vector<float> *m_hits_eta = nullptr;
  std::vector<float> *m_hits_phi = nullptr;
  std::vector<float> *m_hits_rho = nullptr;
  std::vector<float> *m_hits_delta = nullptr;
  std::vector<float> *m_hits_energy = nullptr;
  std::vector<float> *m_hits_MCEnergy = nullptr;

  TTree* t_clusters{nullptr};
  std::vector<int> *m_clusters = nullptr;
  std::vector<int> *m_clusters_event = nullptr;
  std::vector<int> *m_clusters_maxLayer = nullptr;
  std::vector<int> *m_clusters_size = nullptr;
  std::vector<int> *m_clusters_totSize = nullptr;
  std::vector<float> *m_clusters_x = nullptr;
  std::vector<float> *m_clusters_y = nullptr;
  std::vector<float> *m_clusters_z = nullptr;
  std::vector<float> *m_clusters_energy = nullptr;
  std::vector<float> *m_clusters_totEnergy = nullptr;
  std::vector<float> *m_clusters_totEnergyHits = nullptr;
  std::vector<float> *m_clusters_MCEnergy = nullptr;

  TTree* t_clhits{nullptr};
  std::vector<int> *m_clhits_event = nullptr;
  std::vector<int> *m_clhits_layer = nullptr;
  std::vector<float> *m_clhits_x = nullptr;
  std::vector<float> *m_clhits_y = nullptr;
  std::vector<float> *m_clhits_z = nullptr;
  std::vector<float> *m_clhits_energy = nullptr;

  std::int32_t evNum;
};

#endif  // CLUE_HISTOGRAMS_H

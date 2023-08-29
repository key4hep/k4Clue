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
#ifndef CLUE_GAUDI_ALGORITHM_WRAPPER_H
#define CLUE_GAUDI_ALGORITHM_WRAPPER_H

#include <GaudiAlg/GaudiAlgorithm.h>

// FWCore
#include <k4FWCore/DataHandle.h>

#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include "CLUECalorimeterHit.h"

class ClueGaudiAlgorithmWrapper : public GaudiAlgorithm {
public:
  explicit ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc);
  virtual ~ClueGaudiAlgorithmWrapper() = default;
  virtual StatusCode execute() override final;
  virtual StatusCode finalize() override final;
  virtual StatusCode initialize() override final;

  void fillCLUEPoints(std::vector<clue::CLUECalorimeterHit>& clue_hits);
  std::map<int, std::vector<int> > runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits, 
                                           bool isBarrel);
  void cleanCLUEPoints();
  void fillFinalClusters(std::vector<clue::CLUECalorimeterHit>& clue_hits,
                         const std::map<int, std::vector<int> > clusterMap, 
                         edm4hep::ClusterCollection* clusters);
  void calculatePosition(edm4hep::MutableCluster* cluster) ;
  void transformClustersInCaloHits(edm4hep::ClusterCollection* clusters,
                                 edm4hep::CalorimeterHitCollection* caloHits);

  private:
  // Parameters in input
  std::string EBCaloCollectionName;
  std::string EECaloCollectionName;
  const edm4hep::CalorimeterHitCollection* EB_calo_coll; 
  const edm4hep::CalorimeterHitCollection* EE_calo_coll;
  float dc;
  float rhoc;
  float outlierDeltaFactor;

  // CLUE points
  clue::CLUECalorimeterHitCollection clue_hit_coll;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> r;
  std::vector<int> layer;
  std::vector<float> weight;

  // PODIO data service
  ServiceHandle<IDataProviderSvc> m_eventDataSvc;
  PodioLegacyDataSvc* m_podioDataSvc;

  // Collections in output
  DataHandle<edm4hep::CalorimeterHitCollection> caloHitsHandle{"CLUEClustersAsHits", Gaudi::DataHandle::Writer, this};
  DataHandle<edm4hep::ClusterCollection> clustersHandle{"CLUEClusters", Gaudi::DataHandle::Writer, this};

};

#endif

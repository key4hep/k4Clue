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
#ifndef CLUE_GAUDI_ALGORITHM_WRAPPER_H
#define CLUE_GAUDI_ALGORITHM_WRAPPER_H

#include <Gaudi/Algorithm.h>

// FWCore
#include "k4FWCore/DataHandle.h"
#include "k4FWCore/MetaDataHandle.h"

#include "CLUEAlgo.h"
#include "CLUECalorimeterHit.h"
#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/Constants.h>

class ClueGaudiAlgorithmWrapper : public Gaudi::Algorithm {
public:
  explicit ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc);
  virtual ~ClueGaudiAlgorithmWrapper() = default;
  virtual StatusCode execute(const EventContext&) const override final;
  virtual StatusCode finalize() override final;
  virtual StatusCode initialize() override final;

  // Timing analysis
  void exclude_stats_outliers(std::vector<float>& v);
  std::pair<float, float> stats(const std::vector<float>& v);
  void printTimingReport(std::vector<float>& vals, int repeats, const std::string label);

  void fillCLUEPoints(std::vector<clue::CLUECalorimeterHit>& clue_hits) const;
  std::map<int, std::vector<int>> runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits, bool isBarrel) const;
  void cleanCLUEPoints() const;
  void fillFinalClusters(std::vector<clue::CLUECalorimeterHit>& clue_hits,
                         const std::map<int, std::vector<int>> clusterMap, edm4hep::ClusterCollection* clusters) const;
  void calculatePosition(edm4hep::MutableCluster* cluster) const;
  void transformClustersInCaloHits(edm4hep::ClusterCollection* clusters,
                                   edm4hep::CalorimeterHitCollection* caloHits) const;

private:
  // Parameters in input
  mutable const edm4hep::CalorimeterHitCollection* EB_calo_coll;
  mutable const edm4hep::CalorimeterHitCollection* EE_calo_coll;
  float dc;
  float rhoc;
  float outlierDeltaFactor;

  // CLUE points
  mutable clue::CLUECalorimeterHitCollection clue_hit_coll;
  mutable std::vector<float> x;
  mutable std::vector<float> y;
  mutable std::vector<float> r;
  mutable std::vector<int> layer;
  mutable std::vector<float> weight;

  // Handle to read the calo cells and their cellID
  mutable DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle{"BarrelInputHits", Gaudi::DataHandle::Reader,
                                                                       this};
  mutable DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle{"EndcapInputHits", Gaudi::DataHandle::Reader,
                                                                       this};
  MetaDataHandle<std::string> cellIDHandle{EB_calo_handle, edm4hep::labels::CellIDEncoding, Gaudi::DataHandle::Reader};

  // CLUE Algo
  mutable CLICdetBarrelCLUEAlgo clueAlgoBarrel_;
  mutable CLICdetEndcapCLUEAlgo clueAlgoEndcap_;

  // Collections in output
  mutable DataHandle<edm4hep::CalorimeterHitCollection> caloHitsHandle{"CLUEClustersAsHits", Gaudi::DataHandle::Writer,
                                                                       this};
  mutable DataHandle<edm4hep::ClusterCollection> clustersHandle{"CLUEClusters", Gaudi::DataHandle::Writer, this};
};

#endif

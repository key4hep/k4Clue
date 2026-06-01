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

#include "Gaudi/Property.h"
#include "k4FWCore/MetadataUtils.h"
#include "k4FWCore/Transformer.h"

#include "CLUECalorimeterHit.h"
#include "CLUEstering/CLUEstering.hpp"
#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/Constants.h>

#include <string>

using CaloHitColl = edm4hep::CalorimeterHitCollection;
using ClusterColl = edm4hep::ClusterCollection;

using retType = std::tuple<ClusterColl, CaloHitColl>;

template <uint8_t nDim>
struct ClueGaudiAlgorithmWrapper final
    : k4FWCore::MultiTransformer<retType(const std::vector<const CaloHitColl*>& calo_coll)> {

  ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc)
      : MultiTransformer(name, svcLoc,
                         {
                             KeyValues("CaloHitsCollections", {"ECALBarrel", "ECALEndcap"}),
                         },
                         {
                             KeyValue("OutputClusters", "CLUEClusters"),
                             KeyValue("OutputClustersAsHits", "CLUEClustersAsHits"),
                         }) {}

  retType operator()(const std::vector<const CaloHitColl*>& calo_coll) const override;

  StatusCode initialize() override;
  StatusCode finalize() override;

  // Timing analysis
  void exclude_stats_outliers(std::vector<float>& v);
  std::pair<float, float> stats(const std::vector<float>& v);
  void printTimingReport(std::vector<float>& vals, int repeats, const std::string label);

  clue::PointsHost<nDim> fillCLUEPoints(const std::vector<clue::CLUECalorimeterHit>& clue_hits, float* floatBuffer,
                                        int* intBuffer) const;
  clue::AssociationMapHost runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits, const uint32_t offset = 0) const;

  void fillFinalClusters(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                         clue::AssociationMapHost const& clusterMap, ClusterColl& clusters,
                         const std::vector<const CaloHitColl*>& calo_coll) const;
  void fillFinalClustersPerLayer(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                                 clue::AssociationMapHost const& clusterMap, ClusterColl& clusters,
                                 const std::vector<const CaloHitColl*>& calo_coll) const;
  void calculatePosition(edm4hep::MutableCluster* cluster) const;
  void transformClustersInCaloHits(ClusterColl& clusters, CaloHitColl& caloHits) const;

  enum class Coordinate { Cartesian, Polar };

  enum class Strategy { PerCollection, MergeCollections, PerDetectorRegion };

private:
  // Total amount of EE+ and EE- layers (80)
  int m_maxLayerPerSide = 40;

  // CLUE Algo
  mutable std::optional<clue::Clusterer<nDim>> m_clueAlgo;
  mutable std::optional<clue::Queue> m_queue;

  Gaudi::Property<float> m_dc{this, "CriticalDistance", 1.0f, "Distance used to compute the local density of a point"};

  Gaudi::Property<float> m_rhoc{this, "MinLocalDensity", 1.0f,
                                "Minimum energy density of a point to not be considered an outlier"};

  Gaudi::Property<float> m_dm{this, "FollowerDistance", -1.0f,
                              "Critical distance for follower search and cluster expansion"};

  Gaudi::Property<float> m_seed_dc{this, "SeedCriticalDistance", -1.0f,
                                   "Distance used to compute the local density of a point"};

  Gaudi::Property<int> m_pointsPerBin{this, "PointsPerBin", 10,
                                      "Average number of points that are to be found inside a bin"};

  Gaudi::Property<std::string> m_CLUECaloHitCollName{this, "CLUEHitCollName", "CLUECalorimeterHitCollection",
                                                     "Name of the collection of CLUE calorimeter hits"};
  Gaudi::Property<bool> m_saveClustersAsHits{this, "SaveClustersAsHits", false,
                                             "Whether to save clusters as hits in addition to regular clusters"};

  Gaudi::Property<std::string> m_strategyName{this, "strategy", "MergeCollections",
                                              "strategy to treat different collections"};
  Strategy m_strategy;

  Gaudi::Property<std::string> m_coordinateName{this, "coordinate", "Cartesian",
                                                "coordinates to use to cluster points"};
  Coordinate m_coordinate;
};

#endif

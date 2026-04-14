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
#ifndef CLUE_THETAPHI_GAUDI_ALGORITHM_WRAPPER_H
#define CLUE_THETAPHI_GAUDI_ALGORITHM_WRAPPER_H

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
using ClueToCaloHitMap = std::map<const clue::CLUECalorimeterHit*, std::pair<unsigned int, unsigned int>>; // collecion index, hit index

struct ClueThetaPhiGaudiAlgorithmWrapper final
    : k4FWCore::MultiTransformer<std::tuple<ClusterColl>(const std::vector<const CaloHitColl*>&)> {

  ClueThetaPhiGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc)
      : MultiTransformer(name, svcLoc,
                         {
                             KeyValues("CaloHitsCollections", {""}),
                         },
                         {
                             KeyValue("OutputClusters", "CLUEClusters"),
                         }) {}

  std::tuple<ClusterColl> operator()(const std::vector<const CaloHitColl*>& calo_colls) const override;

  StatusCode initialize() override;
  StatusCode finalize() override;

  clue::PointsHost<2> fillCLUEPoints(const std::vector<clue::CLUECalorimeterHit>& clue_hits, float* floatBuffer,
                                     int* intBuffer) const;
  std::vector<std::vector<int>> runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits) const;

  void fillFinalClusters(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                         std::vector<std::vector<int>> const& clusterMap,
                         ClueToCaloHitMap const& clueToCaloHitMap,
                         ClusterColl& clusters,
                         const std::vector<const CaloHitColl*>& calo_colls) const;
  void calculatePosition(edm4hep::MutableCluster* cluster) const;

private:
  // CLUE Algo for 2D
  mutable std::optional<clue::Queue> m_queue;

  Gaudi::Property<float> m_dc{this, "CriticalDistance", 1.0f, "Distance used to compute the local density of a point"};

  Gaudi::Property<float> m_rhoc{this, "MinLocalDensity", 1.0f,
                                "Minimum energy density of a point to not be considered an outlier"};

  Gaudi::Property<float> m_dm{this, "FollowerDistance", 1.0f,
                              "Critical distance for follower search and cluster expansion"};

  Gaudi::Property<float> m_seed_dc{this, "SeedCriticalDistance", -1.0f,
                                   "Minimum distance for a point to be considered as a seed. Negative value sets it to dc"};

  Gaudi::Property<int> m_pointsPerBin{this, "PointsPerBin", 256,
                                      "Average number of points that are to be found inside a bin"};

  Gaudi::Property<std::string> m_CLUECaloHitCollName{this, "CLUEHitCollName", "CLUECalorimeterHitCollection",
                                                     "Name of the collection of CLUE calorimeter hits"};
};

#endif

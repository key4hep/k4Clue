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
#include "ClueThetaPhiGaudiAlgorithmWrapper.h"

#include "CLUEstering/CLUEstering.hpp"
#include <cmath>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
DECLARE_COMPONENT_WITH_ID(ClueThetaPhiGaudiAlgorithmWrapper, "ClueThetaPhiGaudiAlgorithmWrapperCUDA")
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
DECLARE_COMPONENT_WITH_ID(ClueThetaPhiGaudiAlgorithmWrapper, "ClueThetaPhiGaudiAlgorithmWrapperHIP")
#else
DECLARE_COMPONENT_WITH_ID(ClueThetaPhiGaudiAlgorithmWrapper, "ClueThetaPhiGaudiAlgorithmWrapper")
#endif

StatusCode ClueThetaPhiGaudiAlgorithmWrapper::initialize() {
  using Acc = clue::internal::Acc;
  using Dev = clue::Device;
  using Queue = clue::Queue;

  auto const platform = alpaka::Platform<Acc>{};
  Dev const devAcc(alpaka::getDevByIdx(platform, 0u));
  m_queue = std::make_optional<Queue>(devAcc);

  info() << "CLUEAlgo will run on device " << alpaka::getName(devAcc) << " with PeriodicEuclideanMetric" << endmsg;
  info() << "Using (theta, phi) coordinates with phi period 2pi" << endmsg;

  return Algorithm::initialize();
}

clue::PointsHost<2>
ClueThetaPhiGaudiAlgorithmWrapper::fillCLUEPoints(const std::vector<clue::CLUECalorimeterHit>& clue_hits,
                                                   float* floatBuffer, int* intBuffer) const {
  size_t nPoints = clue_hits.size();

  for (size_t i = 0; i < nPoints; ++i) {
    // Convert (x, y, z) to (theta, phi)
    float x = clue_hits[i].getPosition().x;
    float y = clue_hits[i].getPosition().y;
    float z = clue_hits[i].getPosition().z;
    
    float r = std::sqrt(x * x + y * y + z * z);
    float theta = (r > 0) ? std::acos(z / r) : 0.0f;  // theta in [0, pi]
    float phi = std::atan2(y, x);                       // phi in [-pi, pi]
    
    // Normalize phi to [0, 2*pi] for periodic distance calculation
    if (phi < 0) {
      phi += 2.0f * M_PI;
    }
    
    floatBuffer[i] = theta;                          // Fill theta coordinates
    floatBuffer[nPoints + i] = phi;                  // Fill phi coordinates
    floatBuffer[nPoints * 2 + i] = clue_hits[i].getEnergy(); // Fill weights
  }

  // Construct and return the PointsSoA object
  return clue::PointsHost<2>(*m_queue, nPoints, floatBuffer, intBuffer);
}

std::vector<std::vector<int>> ClueThetaPhiGaudiAlgorithmWrapper::runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits) const {
  std::vector<std::vector<int>> clueClusters;

  // Fill CLUE inputs
  size_t nPoints = clue_hits.size();
  std::vector<float> floatBuffer(nPoints * 3); // 2D coordinates + weights
  std::vector<int> intBuffer(nPoints * 2);
  auto cluePoints = fillCLUEPoints(clue_hits, floatBuffer.data(), intBuffer.data());

  // Create a new clusterer for each event to avoid state issues
  clue::Clusterer<2> clueAlgo(*m_queue, m_dc, m_rhoc, m_dm, m_seed_dc, m_pointsPerBin);
  clueAlgo.setWrappedCoordinates(0, 1);

  // Run CLUE
  debug() << "Running CLUEAlgo on device " << alpaka::getName(alpaka::getDev(*m_queue)) << endmsg;

  // Set up the periodic metric for phi coordinate
  // periods = {0, phi_period} (theta non-periodic, phi periodic with period = 2*pi)
  std::array<float, 2> periods = {0.0f, 2.0f * M_PI};
  clue::PeriodicEuclideanMetric<2> metric(periods);

  // Run CLUE clustering with periodic metric
  clue::FlatKernel kernel(0.5f);
  clueAlgo.make_clusters(*m_queue, cluePoints, metric, kernel, 1024);

  debug() << "CLUEAlgo finished clustering" << endmsg;

  // Get clusters using the clusters() method from PointsHost
  int32_t nClusters = cluePoints.n_clusters();

  debug() << "Number of clusters found: " << nClusters << endmsg;

  if (nClusters == 0) {
    return clueClusters; // Return empty if no clusters found
  }

  auto assocMap = cluePoints.clusters();
  clueClusters.resize(nClusters);
  
  // Use equal_range to get all points for each cluster
  for (int32_t clusterId = 0; clusterId < nClusters; ++clusterId) {
    auto range = assocMap.equal_range(clusterId);
    for (auto it = range.first; it != range.second; ++it) {
      clueClusters[clusterId].push_back(*it);
    }
  }

  debug() << "Finished running CLUE algorithm" << endmsg;

  // Including CLUE info in cluePoints
  for (int32_t i = 0; i < cluePoints.size(); i++) {
    clue_hits[i].setClusterIndex(cluePoints.clusterIndexes()[i]);
    verbose() << "CLUE Point #" << i << " : (x,y,z) = (" << clue_hits[i].getPosition().x << ","
              << clue_hits[i].getPosition().y << "," << clue_hits[i].getPosition().z << ")";
    if (cluePoints.clusterIndexes()[i] == -1) {
      verbose() << " is outlier" << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::outlier);
    } else {
      verbose() << " is follower of cluster #" << cluePoints.clusterIndexes()[i] << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::follower);
    }
  }

  return clueClusters;
}

void ClueThetaPhiGaudiAlgorithmWrapper::fillFinalClusters(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                                                          std::vector<std::vector<int>> const& clusterMap,
                                                          ClueToCaloHitMap const& clueToCaloHitMap,
                                                          ClusterColl& clusters, const std::vector<const CaloHitColl*>& calo_colls) const {
  for (auto cl : clusterMap) {
    if (cl.empty()) continue;

    auto cluster = clusters.create();
    unsigned int maxEnergyIndex = 0;
    float maxEnergyValue = 0.f;
    for (auto index : cl) {
      const auto& [collIndex, hitIndex] = clueToCaloHitMap.at(&clue_hits[index]);
      const auto* calo_coll = calo_colls[collIndex];
      cluster.addToHits(calo_coll->at(hitIndex));

      if (clue_hits[index].getEnergy() > maxEnergyValue) {
        maxEnergyValue = clue_hits[index].getEnergy();
        maxEnergyIndex = index;
      }
    }

    float energy = 0.f;
    float sumEnergyErrSquared = 0.f;
    std::for_each(cluster.getHits().begin(), cluster.getHits().end(),
                  [&energy, &sumEnergyErrSquared](edm4hep::CalorimeterHit elem) {
                    energy += elem.getEnergy();
                    sumEnergyErrSquared += pow(elem.getEnergyError() / (1. * elem.getEnergy()), 2);
                  });
    cluster.setEnergy(energy);
    cluster.setEnergyError(std::sqrt(sumEnergyErrSquared));

    calculatePosition(&cluster);

    cluster.setType(clue_hits[maxEnergyIndex].getType());
  }

  return;
}

void ClueThetaPhiGaudiAlgorithmWrapper::calculatePosition(edm4hep::MutableCluster* cluster) const {
  float total_weight = cluster->getEnergy();

  if (total_weight <= 0)
    warning() << "Zero energy in the cluster" << endmsg;

  float total_weight_log = 0.f;
  float x_log = 0.f;
  float y_log = 0.f;
  float z_log = 0.f;
  float error = 0.f;
  float thresholdW0_ = 2.9f; // Min percentage of energy to contribute to the log-reweight position

  for (size_t i = 0; i < cluster->hits_size(); i++) {
    float rhEnergy = cluster->getHits(i).getEnergy();
    float Wi = std::max(thresholdW0_ - std::log(rhEnergy / total_weight), 0.f);
    x_log += cluster->getHits(i).getPosition().x * Wi;
    y_log += cluster->getHits(i).getPosition().y * Wi;
    z_log += cluster->getHits(i).getPosition().z * Wi;
    total_weight_log += Wi;
    error = +1.f / Wi;
  }

  if (total_weight_log != 0.) {
    float inv_tot_weight_log = 1.f / total_weight_log;
    cluster->setPosition({x_log * inv_tot_weight_log, y_log * inv_tot_weight_log, z_log * inv_tot_weight_log});
    cluster->setPositionError({error, 0.f, error, 0.f, 0.f, error});
  }

  return;
}

std::tuple<ClusterColl> ClueThetaPhiGaudiAlgorithmWrapper::operator()(const std::vector<const CaloHitColl*>& calo_colls) const {
  // Output CLUE clusters
  auto finalClusters = ClusterColl();

  // Output CLUE calo hits
  clue::CLUECalorimeterHitCollection clue_hit_coll;
  // map to store the association between CLUE hits and original CaloHits
  ClueToCaloHitMap clueToCaloHitMap; // collecion index, hit index

  // Reserve capacity upfront to prevent vector reallocation, which would
  // invalidate the raw pointers used as keys in clueToCaloHitMap.
  size_t totalHits = 0;
  for (const auto* calo_coll : calo_colls) {
    totalHits += calo_coll->size();
  }
  clue_hit_coll.vect.reserve(totalHits);

  // Fill CLUECaloHits
  for (unsigned int collIndex = 0; collIndex < calo_colls.size(); ++collIndex) {
    const auto* calo_coll = calo_colls[collIndex];

    for (unsigned int hitIndex = 0; hitIndex < calo_coll->size(); ++hitIndex) {
      const auto& calo_hit = calo_coll->at(hitIndex);
      clue::CLUECalorimeterHit clue_hit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::single_region, 0);
      clue_hit_coll.vect.push_back(clue_hit);
      clueToCaloHitMap[&clue_hit_coll.vect.back()] = {collIndex, hitIndex};
    }
  }

  debug() << "Filled " << clue_hit_coll.vect.size() << " CLUE hits." << endmsg;

  // Run CLUE
  if (!clue_hit_coll.vect.empty()) {
    auto clueClusters = runAlgo(clue_hit_coll.vect);
    info() << "Produced " << clueClusters.size() << " clusters" << endmsg;

    fillFinalClusters(clue_hit_coll.vect, clueClusters, clueToCaloHitMap, finalClusters, calo_colls);
    debug() << "Saved " << finalClusters.size() << " clusters" << endmsg;
  } else {
    info() << "No calorimeter hits to process, skipping CLUE algorithm" << endmsg;
  }

  // Cleaning
  clue_hit_coll.vect.clear();

  return std::make_tuple(std::move(finalClusters));
}

StatusCode ClueThetaPhiGaudiAlgorithmWrapper::finalize() {
  return Algorithm::finalize();
}

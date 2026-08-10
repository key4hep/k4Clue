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
#include "ClueGaudiAlgorithmWrapper.h"

#include "IO_helper.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

#include <k4FWCore/MetadataUtils.h>

using namespace dd4hep;
using namespace DDSegmentation;

constexpr float C_MM_NS_SQUARED = 299.792458f * 299.792458f;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<4>, "ClueGaudiAlgorithmWrapperCUDA4D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<3>, "ClueGaudiAlgorithmWrapperCUDA3D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<2>, "ClueGaudiAlgorithmWrapperCUDA2D")
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<4>, "ClueGaudiAlgorithmWrapperHIP4D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<3>, "ClueGaudiAlgorithmWrapperHIP3D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<2>, "ClueGaudiAlgorithmWrapperHIP2D")
#else
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<4>, "ClueGaudiAlgorithmWrapper4D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<3>, "ClueGaudiAlgorithmWrapper3D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<2>, "ClueGaudiAlgorithmWrapper2D")
#endif

namespace {
// Helper function to compute offsets for merged collections
std::vector<size_t> makeOffsets(const std::vector<const CaloHitColl*>& colls) {
  std::vector<size_t> offsets;
  offsets.reserve(colls.size());
  size_t acc = 0;
  for (const auto& c : colls) {
    offsets.push_back(acc);
    acc += c->size();
  }
  return offsets;
}

// Helper function to resolve global index back to collection and hit index
std::pair<size_t, size_t> resolveIndex(const std::vector<size_t>& offsets, size_t globalIndex) {
  // upper_bound finds the first offset > globalIndex, so the correct coll is the one before
  auto it = std::upper_bound(offsets.begin(), offsets.end(), globalIndex);
  size_t collIdx = std::distance(offsets.begin(), it) - 1;
  return {collIdx, globalIndex - offsets[collIdx]};
};
} // anonymous namespace

template <uint8_t nDim>
StatusCode ClueGaudiAlgorithmWrapper<nDim>::initialize() {
  m_queue = clue::get_queue(0u);

  const auto seeding_distance = (m_seed_dc < 0.f) ? m_dc : m_seed_dc;
  const auto outlier_distance = (m_dm < 0.f) ? m_dc : m_dm;
  auto start = std::chrono::high_resolution_clock::now();
  m_clueAlgo = std::make_optional<clue::Clusterer<nDim>>(*m_queue, m_dc, m_rhoc, outlier_distance, seeding_distance,
                                                         m_pointsPerBin);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  debug() << "ClueGaudiAlgorithmWrapper: Set up time: " << elapsed.count() * 1000 << " ms" << endmsg;
  info() << "CLUEAlgo will run on device " << alpaka::getName(alpaka::getDev(*m_queue)) << " and params " << m_dc
         << ", " << m_rhoc << ", " << outlier_distance << ", " << seeding_distance << endmsg;

  if (m_strategyName == "PerDetectorRegion") {
    m_strategy = Strategy::PerDetectorRegion;
  } else if (m_strategyName == "MergeCollections") {
    m_strategy = Strategy::MergeCollections;
  } else if (m_strategyName == "PerCollection") {
    m_strategy = Strategy::PerCollection;
  } else {
    error() << "Unknown strategy: " << m_strategyName << endmsg;
    return StatusCode::FAILURE;
  }

  if (m_coordinateName == "Cartesian") {
    m_coordinate = Coordinate::Cartesian;
  } else if (m_coordinateName == "Polar") {
    m_coordinate = Coordinate::Polar;
    // set periodic coordinates for CLUE algo
    if (nDim == 4) {
      // TODO: Implement custom metric weighted and periodic at the same time
      error() << "Polar coordinates not yet supported for 4D clustering" << endmsg;
      return StatusCode::FAILURE;
    }
    std::vector<uint8_t> coord(nDim, 0);
    coord[1] = 1; // set phi coordinate as periodic
    m_clueAlgo->setWrappedCoordinates(coord);
  } else {
    error() << "Unknown coordinate: " << m_coordinateName << endmsg;
    return StatusCode::FAILURE;
  }

  // Add CellIDEncodingString to CLUE clusters and CLUE calo hits
  // Get collection metadata cellID which is valid for both EB and EE
  const std::string cellIDstr =
      k4FWCore::getCellIDEncoding(inputLocations("CaloHitsCollections")[0], this).value_or("");
  for (auto i = 0u; i < outputLocationsSize(); ++i)
    k4FWCore::putCellIDEncoding(outputLocations(i)[0], cellIDstr, this);

  return Algorithm::initialize();
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::exclude_stats_outliers(std::vector<float>& v) {
  if (v.size() < 2)
    return;
  float mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum_sq_diff = std::accumulate(v.begin(), v.end(), 0.0,
                                      [mean](float acc, float val) { return acc + (val - mean) * (val - mean); });
  float stddev = std::sqrt(sum_sq_diff / (v.size() - 1));
  if (stddev == 0.f)
    return;
  std::cout << "Sigma cut outliers: " << stddev << std::endl;
  float z_score_threshold = 3.0;
  v.erase(std::remove_if(v.begin(), v.end(),
                         [mean, stddev, z_score_threshold](float val) {
                           float z_score = std::abs(val - mean) / stddev;
                           return z_score > z_score_threshold;
                         }),
          v.end());
}

template <uint8_t nDim>
std::pair<float, float> ClueGaudiAlgorithmWrapper<nDim>::stats(const std::vector<float>& v) {
  if (v.empty())
    return {0.f, 0.f};

  float m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum =
      std::accumulate(v.begin(), v.end(), 0.0, [m](float acc, float val) { return acc + (val - m) * (val - m); });
  auto den = v.size() > 1 ? (v.size() - 1) : v.size();
  return {m, std::sqrt(sum / den)};
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::printTimingReport(std::vector<float>& vals, int repeats,
                                                        const std::string label) {
  int precision = 2;
  exclude_stats_outliers(vals);
  auto [mean, sigma] = stats(vals);
  std::cout << label << " 1 outliers(" << repeats << "/" << vals.size() << ") " << std::fixed
            << std::setprecision(precision) << mean << " +/- " << sigma << " [ms]" << std::endl;
  exclude_stats_outliers(vals);
  auto [mean2, sigma2] = stats(vals);
  std::cout << label << " 2 outliers(" << repeats << "/" << vals.size() << ") " << std::fixed
            << std::setprecision(precision) << mean2 << " +/- " << sigma2 << " [ms]" << std::endl;
}

template <uint8_t nDim>
clue::PointsHost<nDim>
ClueGaudiAlgorithmWrapper<nDim>::fillCLUEPoints(const std::vector<clue::CLUECalorimeterHit>& clue_hits,
                                                float* floatBuffer, int* intBuffer) const {
  size_t nPoints = clue_hits.size();

  if (m_coordinate == Coordinate::Cartesian) {
    for (size_t i = 0; i < nPoints; ++i) {
      const auto& position = clue_hits[i].getPosition();
      floatBuffer[i] = position.x;           // Fill x coordinates
      floatBuffer[nPoints + i] = position.y; // Fill y coordinates
      if constexpr (nDim >= 3)
        floatBuffer[nPoints * 2 + i] = position.z; // Fill z coordinates
      if constexpr (nDim >= 4)
        floatBuffer[nPoints * 3 + i] = clue_hits[i].getTime();    // Fill time coordinates
      floatBuffer[nPoints * nDim + i] = clue_hits[i].getEnergy(); // Fill weights
    }
  } else if (m_coordinate == Coordinate::Polar) {
    for (size_t i = 0; i < nPoints; ++i) {
      float phi = clue_hits[i].getPhi();
      // Normalize phi to [0, 2*pi] for periodic distance calculation
      // (clue::PeriodicEuclideanMetric only works with positive periods)
      if (phi < 0) {
        phi += 2.0f * M_PI;
      }

      floatBuffer[i] = clue_hits[i].getTheta(); // Fill theta coordinates
      floatBuffer[nPoints + i] = phi;           // Fill phi coordinates
      if constexpr (nDim >= 3)
        floatBuffer[nPoints * 2 + i] = clue_hits[i].getPosition().z; // Fill z coordinates
      floatBuffer[nPoints * nDim + i] = clue_hits[i].getEnergy();    // Fill weights
    }
  } // if Cartesian or Polar (else should not happen due to checks in initialize())

  // Construct and return the PointsSoA object
  return clue::PointsHost<nDim>(*m_queue, nPoints, floatBuffer, intBuffer);
}

template <uint8_t nDim>
clue::AssociationMapHost ClueGaudiAlgorithmWrapper<nDim>::runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits,
                                                                  const uint32_t offset) const {
  // Fill CLUE inputs
  size_t nPoints = clue_hits.size();
  std::vector<float> floatBuffer(nPoints * (nDim + 1));
  std::vector<int> intBuffer(nPoints * 2);
  auto cluePoints = fillCLUEPoints(clue_hits, floatBuffer.data(), intBuffer.data());

  // Run CLUE
  debug() << "Running CLUEAlgo on device " << alpaka::getName(alpaka::getDev(*m_queue)) << " in " << (uint16_t)nDim
          << "D" << endmsg;

  // measure excution time of make_clusters
  auto start = std::chrono::high_resolution_clock::now();
  if (m_coordinate == Coordinate::Cartesian) {
    if constexpr (nDim == 4) {
      auto metric = clue::metrics::WeightedEuclidean<nDim>(1.f, 1.f, 1.f, C_MM_NS_SQUARED);
      m_clueAlgo->make_clusters(*m_queue, cluePoints, metric);
    } else {
      m_clueAlgo->make_clusters(*m_queue, cluePoints);
    }
  } else if (m_coordinate == Coordinate::Polar) {
    std::array<float, nDim> periods{}; // zero-initialize all to non-periodic
    periods[1] = 2.0f * M_PI;          // set phi coordinate as periodic
    clue::PeriodicEuclideanMetric<nDim> metric(periods);
    m_clueAlgo->make_clusters(*m_queue, cluePoints, metric);
  } // if Cartesian or Polar (else should not happen due to checks in initialize())

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  info() << "ClueGaudiAlgorithmWrapper: Elapsed time: " << elapsed.count() * 1000 << " ms" << endmsg;

  auto clueClusters = m_clueAlgo->getClusters(cluePoints);

  debug() << "Finished running CLUE algorithm" << endmsg;

  // Including CLUE info in cluePoints
  const auto clusterIndexes = cluePoints.clusterIndexes();
  for (int32_t i = 0; i < cluePoints.size(); i++) {
    // offset is 0 for the barrel and is the number of clusters in the barrel for the endcap
    clue_hits[i].setClusterIndex(clusterIndexes[i] + offset);
    verbose() << "CLUE Point #" << i << " : (x,y,z) = (" << clue_hits[i].getPosition().x << ","
              << clue_hits[i].getPosition().y << "," << clue_hits[i].getPosition().z << ")";
    if (clusterIndexes[i] == -1) {
      verbose() << " is outlier" << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::outlier);
    } else {
      verbose() << " is follower of cluster #" << clusterIndexes[i] << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::follower);
    }
  } // for cluePoints

  return clueClusters;
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::fillFinalClusters(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                                                        clue::AssociationMapHost const& clusterMap,
                                                        ClusterColl& clusters,
                                                        const std::vector<const CaloHitColl*>& calo_coll) const {
  // Precompute cumulative offsets once
  const auto collOffsets = makeOffsets(calo_coll);

  for (auto cl = 0u; cl < clusterMap.size(); ++cl) {
    if (clusterMap.empty(cl)) // check if there are elements associated with index cl
      continue;

    auto cluster = clusters.create();
    size_t maxEnergyIndex = 0;
    float maxEnergyValue = 0.f;
    float energy = 0.f;
    float sumEnergyErrSquared = 0.f;
    bool hasMaxEnergy = false;
    for (auto index : clusterMap[cl]) {
      auto [collIdx, localIdx] = resolveIndex(collOffsets, index);
      cluster.addToHits(calo_coll[collIdx]->at(localIdx));

      float hitEne = clue_hits[index].getEnergy();
      if (!hasMaxEnergy || hitEne > maxEnergyValue) {
        maxEnergyValue = hitEne;
        maxEnergyIndex = index;
        hasMaxEnergy = true;
      }
      energy += hitEne;
      const float hitEnergyError = clue_hits[index].getEnergyError();
      sumEnergyErrSquared += hitEnergyError * hitEnergyError;
    }
    cluster.setEnergy(energy);
    cluster.setEnergyError(std::sqrt(sumEnergyErrSquared));

    calculatePosition(&cluster);

    cluster.setType(clue_hits[maxEnergyIndex].getType());
  } // for clusterMap (each cluster)

  return;
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::fillFinalClustersPerLayer(
    std::vector<clue::CLUECalorimeterHit> const& clue_hits, clue::AssociationMapHost const& clusterMap,
    ClusterColl& clusters, const std::vector<const CaloHitColl*>& calo_coll) const {
  if constexpr (nDim == 2) {
    // Precompute cumulative offsets once
    const auto collOffsets = makeOffsets(calo_coll);

    for (auto cl = 0u; cl < clusterMap.size(); ++cl) {
      std::vector<std::vector<int>> clustersLayer(m_maxLayerPerSide * 2);
      for (auto index : clusterMap[cl]) {
        clustersLayer[clue_hits[index].getLayer()].push_back(index);
      }

      for (auto clLay : clustersLayer) {
        if (clLay.empty())
          continue;
        auto cluster = clusters.create();
        size_t maxEnergyIndex = 0;
        float maxEnergyValue = 0.f;
        float energy = 0.f;
        float sumEnergyErrSquared = 0.f;
        bool hasMaxEnergy = false;
        for (auto index : clLay) {
          auto [collIdx, localIdx] = resolveIndex(collOffsets, index);
          cluster.addToHits(calo_coll[collIdx]->at(localIdx));

          float hitEne = clue_hits[index].getEnergy();
          if (!hasMaxEnergy || hitEne > maxEnergyValue) {
            maxEnergyValue = hitEne;
            maxEnergyIndex = index;
            hasMaxEnergy = true;
          }
          energy += hitEne;
          const float hitEnergyError = clue_hits[index].getEnergyError();
          sumEnergyErrSquared += hitEnergyError * hitEnergyError;
        }
        cluster.setEnergy(energy);
        cluster.setEnergyError(sqrt(sumEnergyErrSquared));

        calculatePosition(&cluster);

        cluster.setType(clue_hits[maxEnergyIndex].getType());
      } // for each layer
    } // for clusterMap (each cluster)
  } else {
    fillFinalClusters(clue_hits, clusterMap, clusters, calo_coll);
  }

  return;
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::calculatePosition(edm4hep::MutableCluster* cluster) const {
  float total_weight = cluster->getEnergy();
  if (total_weight <= 0) {
    warning() << "Zero energy in the cluster" << endmsg;
    return;
  }

  // Logarithmic weighting: hits below exp(-W0) of the cluster energy get zero weight and drop out.
  const float w0 = m_logWeightW0;

  const size_t nHits = cluster->hits_size();
  std::vector<float> weights(nHits, 0.f);

  float total_weight_log = 0.f;
  float x_log = 0.f;
  float y_log = 0.f;
  float z_log = 0.f;

  // First pass: the log weights and the weighted barycentre.
  for (size_t i = 0; i < nHits; i++) {
    float rhEnergy = cluster->getHits(i).getEnergy();
    if (rhEnergy <= 0.f)
      continue;

    float Wi = std::max(w0 + std::log(rhEnergy / total_weight), 0.f);
    if (Wi <= 0.f)
      continue;

    weights[i] = Wi;
    x_log += cluster->getHits(i).getPosition().x * Wi;
    y_log += cluster->getHits(i).getPosition().y * Wi;
    z_log += cluster->getHits(i).getPosition().z * Wi;
    total_weight_log += Wi;
  }

  if (total_weight_log <= 0.f) {
    // Every hit is at or below exp(-W0) of the cluster energy, so all of them were rejected.
    // Only reachable for many hits of comparable energy, i.e. LogWeightW0 is too tight.
    warning() << "All " << nHits << " hits of a cluster of energy " << total_weight
              << " fall below the LogWeightW0 = " << w0 << " cut (exp(-W0) = " << std::exp(-w0)
              << " of the cluster energy): leaving its position and position error unset" << endmsg;
    return;
  }

  const float inv_tot_weight_log = 1.f / total_weight_log;
  const float x = x_log * inv_tot_weight_log;
  const float y = y_log * inv_tot_weight_log;
  const float z = z_log * inv_tot_weight_log;
  cluster->setPosition({x, y, z});

  // Second pass: the covariance of the hit positions about that barycentre, using the same log
  // weights.  The position error has to be a squared length, so the previous sum of 1/Wi -- which
  // is dimensionless -- could not be one.
  float cxx = 0.f, cxy = 0.f, cyy = 0.f, cxz = 0.f, cyz = 0.f, czz = 0.f;
  for (size_t i = 0; i < nHits; i++) {
    const float Wi = weights[i];
    if (Wi <= 0.f)
      continue;

    const auto pos = cluster->getHits(i).getPosition();
    const float dx = pos.x - x;
    const float dy = pos.y - y;
    const float dz = pos.z - z;

    cxx += Wi * dx * dx;
    cyy += Wi * dy * dy;
    czz += Wi * dz * dz;
    cxy += Wi * dx * dy;
    cxz += Wi * dx * dz;
    cyz += Wi * dy * dz;
  }

  // edm4hep packs the covariance as the lower triangle: {xx, xy, yy, xz, yz, zz}
  cluster->setPositionError({cxx * inv_tot_weight_log, cxy * inv_tot_weight_log, cyy * inv_tot_weight_log,
                             cxz * inv_tot_weight_log, cyz * inv_tot_weight_log, czz * inv_tot_weight_log});

  return;
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::transformClustersInCaloHits(ClusterColl& clusters, CaloHitColl& caloHits) const {
  float time = 0.f;
  float maxEnergy = 0.f;
  std::uint64_t maxEnergyCellID = 0;

  for (auto cl : clusters) {
    auto caloHit = caloHits.create();
    caloHit.setEnergy(cl.getEnergy());
    caloHit.setEnergyError(cl.getEnergyError());
    caloHit.setPosition(cl.getPosition());
    caloHit.setType(cl.getType());

    time = 0.0;
    maxEnergy = 0.0;
    maxEnergyCellID = 0;
    for (auto hit : cl.getHits()) {
      time += hit.getTime();
      if (hit.getEnergy() > maxEnergy) {
        maxEnergy = hit.getEnergy();
        maxEnergyCellID = hit.getCellID();
      }
    }

    caloHit.setCellID(maxEnergyCellID);
    caloHit.setTime(time / cl.hits_size());
  }

  return;
}

template <uint8_t nDim>
retType ClueGaudiAlgorithmWrapper<nDim>::operator()(const std::vector<const CaloHitColl*>& calo_coll) const {

  // Output CLUE clusters
  auto finalClusters = ClusterColl();

  // Output CLUE calo hits
  clue::CLUECalorimeterHitCollection clue_hit_coll;

  if (m_strategy == Strategy::MergeCollections) {
    for (const auto& coll : calo_coll) {
      for (const auto& calo_hit : *coll) {
        clue_hit_coll.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone()));
      }
    }
    info() << "Processing " << clue_hit_coll.vect.size() << " caloHits in one pass." << endmsg;

    if (!clue_hit_coll.vect.empty()) {
      auto clueClusters = runAlgo(clue_hit_coll.vect);
      info() << "Produced " << clueClusters.size() << " clusters" << endmsg;

      fillFinalClusters(clue_hit_coll.vect, clueClusters, finalClusters, calo_coll);
      debug() << "Saved " << finalClusters.size() << " clusters in total" << endmsg;
    } else {
      info() << "No calorimeter hits to process, skipping CLUE algorithm" << endmsg;
    }
  } else if (m_strategy == Strategy::PerCollection) {
    uint32_t offset = 0;
    int collIndex = 0;
    const std::vector<std::string> ClusterCollectionsNames = inputLocations("CaloHitsCollections");
    for (const auto& coll : calo_coll) {
      clue::CLUECalorimeterHitCollection clue_hit_coll_tmp;
      const std::vector<const CaloHitColl*> currentCaloColl{coll};
      for (const auto& calo_hit : *coll) {
        clue_hit_coll_tmp.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone()));
      }
      info() << "Processing " << clue_hit_coll_tmp.vect.size() << " caloHits in collection "
             << ClusterCollectionsNames[collIndex] << "." << endmsg;
      collIndex++;

      if (!clue_hit_coll_tmp.vect.empty()) {
        auto clueClusters = runAlgo(clue_hit_coll_tmp.vect, offset);
        info() << "Produced " << clueClusters.size() << " clusters" << endmsg;

        fillFinalClusters(clue_hit_coll_tmp.vect, clueClusters, finalClusters, currentCaloColl);

        clue_hit_coll.vect.insert(clue_hit_coll.vect.end(), clue_hit_coll_tmp.vect.begin(),
                                  clue_hit_coll_tmp.vect.end());
        offset += static_cast<uint32_t>(clueClusters.size());
      } else {
        info() << "No calorimeter hits to process, skipping CLUE algorithm" << endmsg;
      }
    }

    debug() << "Saved " << finalClusters.size() << " clusters in total" << endmsg;
  } else {
    const std::vector<std::string> ClusterCollectionsNames = inputLocations("CaloHitsCollections");

    // Get collection metadata cellID which is valid for both EB and EE
    const std::string cellIDstr = k4FWCore::getCellIDEncoding(ClusterCollectionsNames[0], this).value_or("");
    const BitFieldCoder bf(cellIDstr);

    // Fill CLUECaloHits per region
    uint32_t offset = 0;
    int collIndex = 0;
    for (const auto& coll : calo_coll) {
      clue::CLUECalorimeterHitCollection clue_hit_coll_tmp;
      const std::vector<const CaloHitColl*> currentCaloColl{coll};
      std::string const& collName = ClusterCollectionsNames[collIndex];
      collIndex++;
      if (collName.find("Barrel") != std::string::npos) {
        for (const auto& calo_hit : *coll) {
          clue_hit_coll_tmp.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(),
                                                                    clue::CLUECalorimeterHit::DetectorRegion::barrel,
                                                                    bf.get(calo_hit.getCellID(), "layer")));
        } // for each calo_hit in Barrel
      } else if (collName.find("Endcap") != std::string::npos) {
        for (const auto& calo_hit : *coll) {
          if (bf.get(calo_hit.getCellID(), "side") < 0 || bf.get(calo_hit.getCellID(), "side") > 1) {
            clue_hit_coll_tmp.vect.push_back(clue::CLUECalorimeterHit(calo_hit.clone(),
                                                                      clue::CLUECalorimeterHit::DetectorRegion::endcap,
                                                                      bf.get(calo_hit.getCellID(), "layer")));
          } else {
            clue_hit_coll_tmp.vect.push_back(
                clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::endcap,
                                         bf.get(calo_hit.getCellID(), "layer") + m_maxLayerPerSide));
          }
        } // for each calo_hit in Endcap
      } else
        throw std::runtime_error("With 'PerDetectorRegion' strategy the collection must be Barrel or Endcap");

      info() << "Processing " << clue_hit_coll_tmp.vect.size() << " caloHits " << collName << "." << endmsg;

      if (!clue_hit_coll_tmp.vect.empty()) {
        auto clueClusters = runAlgo(clue_hit_coll_tmp.vect, offset);
        info() << "Produced " << clueClusters.size() << " clusters" << endmsg;

        fillFinalClustersPerLayer(clue_hit_coll_tmp.vect, clueClusters, finalClusters, currentCaloColl);

        clue_hit_coll.vect.insert(clue_hit_coll.vect.end(), clue_hit_coll_tmp.vect.begin(),
                                  clue_hit_coll_tmp.vect.end());
        offset += static_cast<uint32_t>(clueClusters.size());
      } else {
        info() << "No calorimeter hits to process, skipping CLUE algorithm" << endmsg;
      }

    } // for each collection
    debug() << "Saved " << finalClusters.size() << " clusters in total" << endmsg;
  } // if-else on strategy

  // if configured, save clusters as calo hits as well, in addition to regular clusters
  auto finalCaloHits = CaloHitColl();

  if (m_saveClustersAsHits) {
    debug() << "Saving clusters as calo hits as well, in addition to regular clusters" << endmsg;

    // Save CLUE calo hits
    auto pCHV = std::make_unique<clue::CLUECalorimeterHitCollection>(clue_hit_coll);
    const StatusCode scStatusV = eventSvc()->registerObject("/Event/" + m_CLUECaloHitCollName, pCHV.release());
    if (scStatusV.isFailure())
      throw std::runtime_error("Failed to register " + m_CLUECaloHitCollName);

    debug() << "Saved " << clue_hit_coll.vect.size() << " CLUE calo hits in total. " << endmsg;

    // Save clusters as calo hits and add cellID to them
    transformClustersInCaloHits(finalClusters, finalCaloHits);
    debug() << "Saved " << finalCaloHits.size() << " clusters as calo hits" << endmsg;

  } // if m_saveClustersAsHits

  // Cleaning
  clue_hit_coll.vect.clear();

  return std::make_tuple(std::move(finalClusters), std::move(finalCaloHits));
}

template <uint8_t nDim>
StatusCode ClueGaudiAlgorithmWrapper<nDim>::finalize() {
  return Algorithm::finalize();
}

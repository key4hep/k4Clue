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

using namespace dd4hep;
using namespace DDSegmentation;
using namespace std;

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<3>, "ClueGaudiAlgorithmWrapperCUDA3D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<2>, "ClueGaudiAlgorithmWrapperCUDA2D")
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<3>, "ClueGaudiAlgorithmWrapperHIP3D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<2>, "ClueGaudiAlgorithmWrapperHIP2D")
#else
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<3>, "ClueGaudiAlgorithmWrapper3D")
DECLARE_COMPONENT_WITH_ID(ClueGaudiAlgorithmWrapper<2>, "ClueGaudiAlgorithmWrapper2D")
#endif

template <uint8_t nDim>
ClueGaudiAlgorithmWrapper<nDim>::ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* pSL)
    : Gaudi::Algorithm(name, pSL) {
  declareProperty("BarrelCaloHitsCollection", EB_calo_handle, "Collection for Barrel Calo Hits used in input");
  declareProperty("EndcapCaloHitsCollection", EE_calo_handle, "Collection for Endcap Calo Hits used in input");
  declareProperty("CriticalDistance", dc, "Distance used to compute the local density of a point");
  declareProperty("MinLocalDensity", rhoc, "Minimum energy density of a point to not be considered an outlier");
  declareProperty("FollowerDistance", dm, "Critical distance for follower search and cluster expansion");
  declareProperty("SeedCriticalDistance", seed_dc, "Distance used to compute the local density of a point");
  declareProperty("PointsPerBin", pointsPerBin, "Average number of points that are to be found inside a bin");
  declareProperty("OutClusters", clustersHandle, "Clusters collection (output)");
  declareProperty("OutCaloHits", caloHitsHandle, "Calo hits collection created from Clusters (output)");
}

template <uint8_t nDim>
StatusCode ClueGaudiAlgorithmWrapper<nDim>::initialize() {
  using Acc = clue::internal::Acc;
  using Dev = clue::Device;
  using Queue = clue::Queue;

  auto const platform = alpaka::Platform<Acc>{};
  Dev const devAcc(alpaka::getDevByIdx(platform, 0u));
  queue_ = std::make_optional<Queue>(devAcc);

  auto start = std::chrono::high_resolution_clock::now();
  clueAlgo_ = std::make_optional<clue::Clusterer<nDim>>(*queue_, dc, rhoc, dm, seed_dc, pointsPerBin);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  info() << "ClueGaudiAlgorithmWrapper: Set up time: " << elapsed.count() * 1000 << " ms" << endmsg;

  return Algorithm::initialize();
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::exclude_stats_outliers(std::vector<float>& v) {
  if (v.size() == 1)
    return;
  float mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  float sum_sq_diff = std::accumulate(v.begin(), v.end(), 0.0,
                                      [mean](float acc, float val) { return acc + (val - mean) * (val - mean); });
  float stddev = std::sqrt(sum_sq_diff / (v.size() - 1));
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
  float mean = 0.f;
  float sigma = 0.f;
  exclude_stats_outliers(vals);
  tie(mean, sigma) = stats(vals);
  std::cout << label << " 1 outliers(" << repeats << "/" << vals.size() << ") " << std::fixed
            << std::setprecision(precision) << mean << " +/- " << sigma << " [ms]" << std::endl;
  exclude_stats_outliers(vals);
  tie(mean, sigma) = stats(vals);
  std::cout << label << " 2 outliers(" << repeats << "/" << vals.size() << ") " << std::fixed
            << std::setprecision(precision) << mean << " +/- " << sigma << " [ms]" << std::endl;
}

template <uint8_t nDim>
clue::PointsHost<nDim>
ClueGaudiAlgorithmWrapper<nDim>::fillCLUEPoints(const std::vector<clue::CLUECalorimeterHit>& clue_hits,
                                                float* floatBuffer, int* intBuffer, const bool /*isBarrel*/) const {
  size_t nPoints = clue_hits.size();

  for (size_t i = 0; i < nPoints; ++i) {
    // if (isBarrel) {
    //   floatBuffer[i] = clue_hits[i].getPhi();                  // Fill phi coordinates
    //   floatBuffer[nPoints + i] = clue_hits[i].getPosition().z; // Fill z coordinates
    //   if constexpr (nDim == 3)
    //     floatBuffer[nPoints * 2 + i] = clue_hits[i].getEta();     // Fill eta coordinates
    //   floatBuffer[nPoints * nDim + i] = clue_hits[i].getEnergy(); // Fill weights
    // } else {
    floatBuffer[i] = clue_hits[i].getPosition().x;           // Fill x coordinates
    floatBuffer[nPoints + i] = clue_hits[i].getPosition().y; // Fill y coordinates
    if constexpr (nDim == 3)
      floatBuffer[nPoints * 2 + i] = clue_hits[i].getPosition().z; // Fill z coordinates
    floatBuffer[nPoints * nDim + i] = clue_hits[i].getEnergy();    // Fill weights
    //}
  }

  // Construct and return the PointsSoA object
  return clue::PointsHost<nDim>(*queue_, nPoints, floatBuffer, intBuffer);
}

template <uint8_t nDim>
std::vector<std::vector<int>> ClueGaudiAlgorithmWrapper<nDim>::runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits,
                                                                       const bool isBarrel,
                                                                       const uint32_t offset) const {
  std::vector<std::vector<int>> clueClusters;

  // Fill CLUE inputs
  size_t nPoints = clue_hits.size();
  std::vector<float> floatBuffer(nPoints * (nDim + 1));
  std::vector<int> intBuffer(nPoints * 2);
  auto cluePoints = fillCLUEPoints(clue_hits, floatBuffer.data(), intBuffer.data(), isBarrel);

  // Run CLUE
  info() << "Running CLUEAlgo on device " << alpaka::getName(alpaka::getDev(*queue_)) << endmsg;

  // measure excution time of make_clusters
  clue::FlatKernel kernel(0.5f);
  auto start = std::chrono::high_resolution_clock::now();
  clueAlgo_->make_clusters(*queue_, cluePoints, kernel, 512);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  info() << "ClueGaudiAlgorithmWrapper: Elapsed time: " << elapsed.count() * 1000 << " ms" << endmsg;

  clueClusters = clueAlgo_->getClusters(cluePoints);

  info() << "Finished running CLUE algorithm" << endmsg;

  // Including CLUE info in cluePoints
  for (int32_t i = 0; i < cluePoints.size(); i++) {
    // offset is 0 for the barrel and is the number of clusters in the barrel for the endcap
    clue_hits[i].setClusterIndex(cluePoints.clusterIndexes()[i] + offset);
    verbose() << "CLUE Point #" << i << " : (x,y,z) = (" << clue_hits[i].getPosition().x << ","
              << clue_hits[i].getPosition().y << "," << clue_hits[i].getPosition().z << ")";
    if (cluePoints.isSeed()[i] == 1) {
      verbose() << " is seed" << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::seed);
    } else if (cluePoints.clusterIndexes()[i] == -1) {
      verbose() << " is outlier" << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::outlier);
    } else {
      verbose() << " is follower of cluster #" << cluePoints.clusterIndexes()[i] << endmsg;
      clue_hits[i].setStatus(clue::CLUECalorimeterHit::Status::follower);
    }
  }

  return clueClusters;
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::fillFinalClusters(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                                                        std::vector<std::vector<int>> const& clusterMap,
                                                        edm4hep::ClusterCollection* clusters) const {
  for (auto cl : clusterMap) {
    auto cluster = clusters->create();
    unsigned int maxEnergyIndex = 0;
    float maxEnergyValue = 0.f;
    for (auto index : cl) {
      if (clue_hits[index].inBarrel()) {
        cluster.addToHits(EB_calo_coll->at(index));
      }
      if (clue_hits[index].inEndcap()) {
        cluster.addToHits(EE_calo_coll->at(index));
      }

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
    cluster.setEnergyError(sqrt(sumEnergyErrSquared));

    calculatePosition(&cluster);

    cluster.setType(clue_hits[maxEnergyIndex].getType());
  }
  return;
}

template <>
void ClueGaudiAlgorithmWrapper<2>::fillFinalClusters(std::vector<clue::CLUECalorimeterHit> const& clue_hits,
                                                     std::vector<std::vector<int>> const& clusterMap,
                                                     edm4hep::ClusterCollection* clusters) const {
  for (auto cl : clusterMap) {
    std::vector<std::vector<int>> clustersLayer(maxLayerPerSide * 2);
    for (auto index : cl) {
      clustersLayer[clue_hits[index].getLayer()].push_back(index);
    }
    for (auto clLay : clustersLayer) {
      if (clLay.empty())
        continue;
      auto cluster = clusters->create();
      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;
      for (auto index : cl) {
        if (clue_hits[index].inBarrel()) {
          cluster.addToHits(EB_calo_coll->at(index));
        }
        if (clue_hits[index].inEndcap()) {
          cluster.addToHits(EE_calo_coll->at(index));
        }

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
      cluster.setEnergyError(sqrt(sumEnergyErrSquared));

      calculatePosition(&cluster);

      cluster.setType(clue_hits[maxEnergyIndex].getType());
    }
  }
  return;
}

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::calculatePosition(edm4hep::MutableCluster* cluster) const {
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

template <uint8_t nDim>
void ClueGaudiAlgorithmWrapper<nDim>::transformClustersInCaloHits(edm4hep::ClusterCollection* clusters,
                                                                  edm4hep::CalorimeterHitCollection* caloHits) const {
  float time = 0.f;
  float maxEnergy = 0.f;
  std::uint64_t maxEnergyCellID = 0;

  for (auto cl : *clusters) {
    auto caloHit = caloHits->create();
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
StatusCode ClueGaudiAlgorithmWrapper<nDim>::execute(const EventContext&) const {
  // Read EB and EE collection
  EB_calo_coll = EB_calo_handle.get();
  EE_calo_coll = EE_calo_handle.get();

  // Get collection metadata cellID which is valid for both EB and EE
  const auto cellIDstr = cellIDHandle.get();
  const BitFieldCoder bf(cellIDstr);

  // Output CLUE clusters
  auto finalClusters = std::make_unique<edm4hep::ClusterCollection>();

  // Output CLUE calo hits
  clue::CLUECalorimeterHitCollection clue_hit_coll_barrel;
  clue::CLUECalorimeterHitCollection clue_hit_coll_endcap;

  debug() << "ClueGaudiAlgorithmWrapper: Total number of calo hits: "
          << int(EB_calo_coll->size() + EE_calo_coll->size()) << endmsg;
  info() << "Processing " << EB_calo_coll->size() << " caloHits in ECAL Barrel." << endmsg;

  // Fill CLUECaloHits in the barrel
  for (const auto& calo_hit : (*EB_calo_coll)) {
    // Cut on a specific layer for noise studies
    // if(bf.get( calo_hit.getCellID(), "layer") == 6){
    clue_hit_coll_barrel.vect.push_back(clue::CLUECalorimeterHit(
        calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::barrel, bf.get(calo_hit.getCellID(), "layer")));
    //}
  }

  // Run CLUE in the barrel
  if (!clue_hit_coll_barrel.vect.empty()) {
    auto clueClustersBarrel = runAlgo(clue_hit_coll_barrel.vect);
    info() << "Produced " << clueClustersBarrel.size() << " clusters in ECAL Barrel" << endmsg;

    clue_hit_coll.vect.insert(clue_hit_coll.vect.end(), clue_hit_coll_barrel.vect.begin(),
                              clue_hit_coll_barrel.vect.end());

    fillFinalClusters(clue_hit_coll_barrel.vect, clueClustersBarrel, finalClusters.get());
    debug() << "Saved " << finalClusters->size() << " clusters using ECAL Barrel hits" << endmsg;
  }
  uint32_t barrelOffset = finalClusters->size();

  info() << "Processing " << EE_calo_coll->size() << " caloHits in ECAL Endcap." << endmsg;

  // Fill CLUECaloHits in the endcap
  for (const auto& calo_hit : (*EE_calo_coll)) {
    if (bf.get(calo_hit.getCellID(), "side") < 0 || bf.get(calo_hit.getCellID(), "side") > 1) {
      clue_hit_coll_endcap.vect.push_back(clue::CLUECalorimeterHit(
          calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::endcap, bf.get(calo_hit.getCellID(), "layer")));
    } else {
      clue_hit_coll_endcap.vect.push_back(
          clue::CLUECalorimeterHit(calo_hit.clone(), clue::CLUECalorimeterHit::DetectorRegion::endcap,
                                   bf.get(calo_hit.getCellID(), "layer") + maxLayerPerSide));
    }
  }

  // Run CLUE in the endcap
  if (!clue_hit_coll_endcap.vect.empty()) {
    auto clueClustersEndcap = runAlgo(clue_hit_coll_endcap.vect, false, barrelOffset);
    info() << "Produced " << clueClustersEndcap.size() << " clusters in ECAL Endcap" << endmsg;

    clue_hit_coll.vect.insert(clue_hit_coll.vect.end(), clue_hit_coll_endcap.vect.begin(),
                              clue_hit_coll_endcap.vect.end());

    fillFinalClusters(clue_hit_coll_endcap.vect, clueClustersEndcap, finalClusters.get());
    debug() << "Saved " << finalClusters->size() - barrelOffset << " clusters using ECAL Endcap hits" << endmsg;
  }

  info() << "Saved " << finalClusters->size() << " CLUE clusters in total." << endmsg;

  // Save CLUE calo hits
  auto pCHV = std::make_unique<clue::CLUECalorimeterHitCollection>(clue_hit_coll);
  const StatusCode scStatusV = eventSvc()->registerObject("/Event/CLUECalorimeterHitCollection", pCHV.release());
  if (scStatusV.isFailure()) {
    error() << "Failed to register CLUECalorimeterHitCollection" << endmsg;
    return StatusCode::FAILURE;
  }
  info() << "Saved " << clue_hit_coll.vect.size() << " CLUE calo hits in total. " << endmsg;

  // Save clusters as calo hits and add cellID to them
  auto finalCaloHits = std::make_unique<edm4hep::CalorimeterHitCollection>();
  transformClustersInCaloHits(finalClusters.get(), finalCaloHits.get());
  info() << "Saved " << finalCaloHits->size() << " clusters as calo hits" << endmsg;

  // Only now can we put the collections into the event store, as nothing needs
  // them any longer
  caloHitsHandle.put(std::move(finalCaloHits));
  clustersHandle.put(std::move(finalClusters));

  // To be fixed in the future:
  // Add CellIDEncodingString to CLUE clusters and CLUE calo hits

  // Cleaning
  clue_hit_coll.vect.clear();

  return StatusCode::SUCCESS;
}

template <uint8_t nDim>
StatusCode ClueGaudiAlgorithmWrapper<nDim>::finalize() {
  return Algorithm::finalize();
}

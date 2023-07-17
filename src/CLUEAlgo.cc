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
#include "CLUEAlgo.h"
#include <array>

template <typename TILES>
void CLUEAlgo_T<TILES>::makeClusters(){

  TILES allLayerTiles;

  // start clustering
  auto start = std::chrono::high_resolution_clock::now();

  prepareDataStructures(allLayerTiles);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  if(verbose_)
    std::cout << "--- prepareDataStructures:     " << elapsed.count() *1000 << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  calculateLocalDensity(allLayerTiles);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  if(verbose_)
    std::cout << "--- calculateLocalDensity:     " << elapsed.count() *1000 << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  calculateDistanceToHigher(allLayerTiles);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  if(verbose_)
    std::cout << "--- calculateDistanceToHigher: " << elapsed.count() *1000 << " ms\n";

  findAndAssignClusters();  

}

template <typename TILES>
std::map<int, std::vector<int> > CLUEAlgo_T<TILES>::getClusters(){
  // cluster all points with same clusterId
  std::map<int, std::vector<int> > clusters; 
  for(unsigned i = 0; i < points_.n; i++) {
    clusters[points_.clusterIndex[i]].push_back(i);
  }
  return clusters;
}

template <typename TILES>
void CLUEAlgo_T<TILES>::prepareDataStructures( TILES & allLayerTiles ){
  for (int i=0; i<points_.n; i++){
    // push index of points into tiles
    allLayerTiles.fill( points_.layer[i], points_.x[i], points_.y[i], points_.x[i]/(1.*points_.r[i]), i );
  }
}

template <typename TILES>
void CLUEAlgo_T<TILES>::calculateLocalDensity( TILES & allLayerTiles ){

  std::array<int,4> search_box = {0, 0, 0, 0};
  auto dc2 = dc_*dc_;

  // loop over all points
  for(unsigned i = 0; i < points_.n; i++) {
    const auto& lt = allLayerTiles[points_.layer[i]];
    float ri = points_.r[i];
    float inv_ri = 1.f/ri;
    float phi_i = points_.x[i]*inv_ri;

    // get search box
    search_box = lt.searchBox(points_.x[i]-dc_, points_.x[i]+dc_, points_.y[i]-dc_, points_.y[i]+dc_);

    if(!TILES::constants_type_t::endcap){
      float dc_phi = dc_*inv_ri;
      search_box = lt.searchBoxPhiZ(phi_i-dc_phi, phi_i+dc_phi, points_.y[i]-dc_, points_.y[i]+dc_);
    }

    // loop over bins in the search box
    for(int xBin = search_box[0]; xBin <= search_box[1]; ++xBin) {
      for(int yBin = search_box[2]; yBin <= search_box[3]; ++yBin) {
  
        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin,yBin);
        if(!TILES::constants_type_t::endcap){
          int phi = (xBin % TILES::constants_type_t::nColumnsPhi);
          binId = lt.getGlobalBinByBinPhi(phi, yBin);
        }
        // get the size of this bin
        int binSize = lt[binId].size();

        // iterate inside this bin
        for (unsigned int binIter = 0; binIter < binSize; binIter++) {
          int j = lt[binId][binIter];
          // query N_{dc_}(i)
          float dist2_ij = TILES::constants_type_t::endcap ?
           distance2(i, j) : distance2(i, j, true, ri);
          if(dist2_ij <= dc2) {
            // sum weights within N_{dc_}(i)
            points_.rho[i] += (i == j ? 1.f : 0.5f) * points_.weight[j];
          }
        } // end of interate inside this bin
      } 
    } // end of loop over bins in search box
  } // end of loop over points

}


template <typename TILES>
void CLUEAlgo_T<TILES>::calculateDistanceToHigher( TILES & allLayerTiles ){
  // loop over all points
  float dm = outlierDeltaFactor_ * dc_;
  for(unsigned i = 0; i < points_.n; i++) {
    // default values of delta and nearest higher for i
    float delta_i = std::numeric_limits<float>::max();
    int nearestHigher_i = -1;
    float xi = points_.x[i];
    float yi = points_.y[i];
    float ri = points_.r[i];
    float inv_ri = 1.f/ri;
    float phi_i = points_.x[i]*inv_ri;
    float rho_i = points_.rho[i];

    //get search box
    const auto& lt = allLayerTiles[points_.layer[i]];
    float dm_phi = dm*inv_ri;
    std::array<int,4> search_box = TILES::constants_type_t::endcap ? 
     lt.searchBox(xi-dm, xi+dm, yi-dm, yi+dm):
     lt.searchBoxPhiZ(phi_i-dm_phi, phi_i+dm_phi, points_.y[i]-dm, points_.y[i]+dm);

    // loop over all bins in the search box
    for(int xBin = search_box[0]; xBin <= search_box[1]; ++xBin) {
      for(int yBin = search_box[2]; yBin <= search_box[3]; ++yBin) {

        // get the id of this bin
        int phi = (xBin % TILES::constants_type_t::nColumnsPhi);
        int binId = TILES::constants_type_t::endcap ?
         lt.getGlobalBinByBin(xBin,yBin):
         lt.getGlobalBinByBinPhi(phi, yBin);

        // get the size of this bin
        int binSize = lt[binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = lt[binId][binIter];
          // query N'_{dm}(i)
          bool foundHigher = (points_.rho[j] > rho_i);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_.rho[j] == rho_i) && (j>i) );
          float dist_ij = TILES::constants_type_t::endcap ?
           distance(i, j) : distance(i, j, true, ri);
          if(foundHigher && dist_ij <= dm) { // definition of N'_{dm}(i)
            // find the nearest point within N'_{dm}(i)
            if (dist_ij < delta_i) {
              // update delta_i and nearestHigher_i
              delta_i = dist_ij;
              nearestHigher_i = j;
            }
          }
        } // end of interate inside this bin
      }
    } // end of loop over bins in search box

    points_.delta[i] = delta_i;
    points_.nearestHigher[i] = nearestHigher_i;
  } // end of loop over points

}

template <typename TILES>
void CLUEAlgo_T<TILES>::findAndAssignClusters(){

  auto start = std::chrono::high_resolution_clock::now();

  std::array<int,TILES::constants_type_t::nLayers> nClustersPerLayer{};

  // find cluster seeds and outlier
  std::vector<int> localStack;
  // loop over all points
  for(unsigned i = 0; i < points_.n; i++) {
    // initialize clusterIndex
    points_.clusterIndex[i] = -1;

    float deltai = points_.delta[i];
    float rhoi = points_.rho[i];

    // determine seed or outlier 
    bool isSeed = (deltai > dc_) and (rhoi >= rhoc_);
    bool isOutlier = (deltai > outlierDeltaFactor_ * dc_) and (rhoi < rhoc_);
    if (isSeed)
      {
	// set isSeed as 1
	points_.isSeed[i] = 1;
	// set cluster id
	points_.clusterIndex[i] = nClustersPerLayer[points_.layer[i]];
	// increment number of clusters
        nClustersPerLayer[points_.layer[i]]++;
	// add seed into local stack
	localStack.push_back(i);
      }
    else if (!isOutlier)
      {
	// register as follower at its nearest higher
	points_.followers[points_.nearestHigher[i]].push_back(i);   
      }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  if(verbose_)
    std::cout << "--- findSeedAndFollowers:      " << elapsed.count() *1000 << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  // expend clusters from seeds
  while (!localStack.empty()) {
    int i = localStack.back();
    auto& followers = points_.followers[i];
    localStack.pop_back();

    // loop over followers
    for(int j : followers){
      // pass id from i to a i's follower
      points_.clusterIndex[j] = points_.clusterIndex[i];
      // push this follower to localStack
      localStack.push_back(j);
    }
  }
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  if(verbose_)
    std::cout << "--- assignClusters:            " << elapsed.count() *1000 << " ms\n";

}

template <typename TILES>
inline float CLUEAlgo_T<TILES>::distance2(int i, int j, bool isPhi, float r ) const {

  // 2-d distance on the layer
  if(points_.layer[i] == points_.layer[j] ) {
    if (isPhi) {
      const float phi_i = points_.x[i]/(points_.r[i]);
      const float phi_j = points_.x[j]/(points_.r[j]);
      const float drphi = r * reco::deltaPhi(phi_i, phi_j);
      const float dy = points_.y[i] - points_.y[j];
      return dy * dy + drphi * drphi;
    } else {
      const float dx = points_.x[i] - points_.x[j];
      const float dy = points_.y[i] - points_.y[j];
      return dx * dx + dy * dy;
    }
  } else {
    return std::numeric_limits<float>::max();
  }

}

template <typename TILES>
inline float CLUEAlgo_T<TILES>::distance(int i, int j, bool isPhi, float r ) const {

  // 2-d distance on the layer
  if(points_.layer[i] == points_.layer[j] ) {
    return std::sqrt(distance2(i, j, isPhi, r));
  } else {
    return std::numeric_limits<float>::max();
  }

}

// explicit template instantiation
template class CLUEAlgo_T<LayerTiles>;
template class CLUEAlgo_T<CLICdetEndcapLayerTiles>;
template class CLUEAlgo_T<CLICdetBarrelLayerTiles>;
template class CLUEAlgo_T<CLDEndcapLayerTiles>;
template class CLUEAlgo_T<CLDBarrelLayerTiles>;
template class CLUEAlgo_T<LArBarrelLayerTiles>;

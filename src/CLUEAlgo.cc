#include "CLUEAlgo.h"

void CLUEAlgo::makeClusters(){
  std::array<LayerTiles, NLAYERS> allLayerTiles;
  // start clustering
  auto start = std::chrono::high_resolution_clock::now();

  prepareDataStructures(allLayerTiles);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "--- prepareDataStructures:     " << elapsed.count() *1000 << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  calculateLocalDensity(allLayerTiles);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- calculateLocalDensity:     " << elapsed.count() *1000 << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  calculateDistanceToHigher(allLayerTiles);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- calculateDistanceToHigher: " << elapsed.count() *1000 << " ms\n";

  findAndAssignClusters();  
}


void CLUEAlgo::prepareDataStructures( std::array<LayerTiles, NLAYERS> & allLayerTiles ){
  for (int i=0; i<points_.n; i++){
    // push index of points into tiles
    allLayerTiles[points_.layer[i]].fill( points_.x[i], points_.y[i], i);
  }
}


void CLUEAlgo::calculateLocalDensity( std::array<LayerTiles, NLAYERS> & allLayerTiles ){

  // loop over all points
  for(unsigned i = 0; i < points_.n; i++) {
    LayerTiles& lt = allLayerTiles[points_.layer[i]];

    // get search box
    std::array<int,4> search_box = lt.searchBox(points_.x[i]-dc_, points_.x[i]+dc_, points_.y[i]-dc_, points_.y[i]+dc_);

    // loop over bins in the search box
    for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
      for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {

        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin,yBin);
        // get the size of this bin
        int binSize = lt[binId].size();
        
        // iterate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = lt[binId][binIter];
          // query N_{dc_}(i)
          float dist_ij = distance(i, j);
          if(dist_ij <= dc_) {
            // sum weights within N_{dc_}(i)
            points_.rho[i] += (i == j ? 1.f : 0.5f) * points_.weight[j];
          }
        } // end of interate inside this bin

      }
    } // end of loop over bins in search box
  } // end of loop over points
}


void CLUEAlgo::calculateDistanceToHigher( std::array<LayerTiles, NLAYERS> & allLayerTiles ){
  // loop over all points
  float dm = outlierDeltaFactor_ * dc_;
  for(unsigned i = 0; i < points_.n; i++) {
    // default values of delta and nearest higher for i
    float delta_i = std::numeric_limits<float>::max();
    int nearestHigher_i = -1;
    float xi = points_.x[i];
    float yi = points_.y[i];
    float rho_i = points_.rho[i];

    //get search box
    LayerTiles& lt = allLayerTiles[points_.layer[i]];
    std::array<int,4> search_box = lt.searchBox(xi-dm, xi+dm, yi-dm, yi+dm);
    
    // loop over all bins in the search box
    for(int xBin = search_box[0]; xBin < search_box[1]+1; ++xBin) {
      for(int yBin = search_box[2]; yBin < search_box[3]+1; ++yBin) {

        // get the id of this bin
        int binId = lt.getGlobalBinByBin(xBin,yBin);
        // get the size of this bin
        int binSize = lt[binId].size();

        // interate inside this bin
        for (int binIter = 0; binIter < binSize; binIter++) {
          int j = lt[binId][binIter];
          // query N'_{dm}(i)
          bool foundHigher = (points_.rho[j] > rho_i);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_.rho[j] == rho_i) && (j>i) );
          float dist_ij = distance(i, j);
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

void CLUEAlgo::findAndAssignClusters(){
  auto start = std::chrono::high_resolution_clock::now();

  int nClusters = 0;

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
	points_.clusterIndex[i] = nClusters;
	// increment number of clusters
	nClusters++;
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
  std::cout << "--- assignClusters:            " << elapsed.count() *1000 << " ms\n";

}

inline float CLUEAlgo::distance(int i, int j) const {

  // 2-d distance on the layer
  if(points_.layer[i] == points_.layer[j] ) {
    const float dx = points_.x[i] - points_.x[j];
    const float dy = points_.y[i] - points_.y[j];
    return std::sqrt(dx * dx + dy * dy);
  } else {
    return std::numeric_limits<float>::max();
  }

}

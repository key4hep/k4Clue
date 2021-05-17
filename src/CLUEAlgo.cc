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
  for(int i = 0; i < points_.n; i++) {
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
  for(int i = 0; i < points_.n; i++) {
    // default values of delta and nearest higher for i
    float maxDelta = std::numeric_limits<float>::max();
    float delta_i = maxDelta;
    int nearestHigher_i = -1;

    LayerTiles& lt = allLayerTiles[points_.layer[i]];

    // get search box 
    std::array<int,4> search_box = lt.searchBox(points_.x[i]-dm, points_.x[i]+dm, points_.y[i]-dm, points_.y[i]+dm);
    
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
          bool foundHigher = (points_.rho[j] > points_.rho[i]);
          // in the rare case where rho is the same, use detid
          foundHigher = foundHigher || ((points_.rho[j] == points_.rho[i]) && (j>i) );
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
  for(int i = 0; i < points_.n; i++) {
    // initialize clusterIndex
    points_.clusterIndex[i] = -1;
    // determine seed or outlier 
    bool isSeed = (points_.delta[i] > dc_) and (points_.rho[i] >= rhoc_);
    bool isOutlier = (points_.delta[i] > outlierDeltaFactor_ * dc_) and (points_.rho[i] < rhoc_);
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
    for( int j : followers){
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

//get an array that tells which hits are seeds, based on their density and assigned cluster
//only works if the function makeClusters() was run first
void CLUEAlgo::infoSeeds()
{
  int noutliers = 0;
  std::vector<int> clusterIdxUsed;
  for(int i = 0; i < points_.n; i++)
    {
      if(points_.clusterIndex[i] == -1) { //no outliers
	noutliers += 1;
	continue;
      }
      if( std::find(clusterIdxUsed.begin(), clusterIdxUsed.end(), points_.clusterIndex[i]) == clusterIdxUsed.end() ) //not found
	{
	  clusterIdxUsed.push_back( points_.clusterIndex[i] );
	  float maxrho = -1.f; //a negative density is lower than any physical hit density
	  int seedIdx = i; //by default the first hit is the seed (always true when the cluster only has one hit)
	  for(int j = i+1; j < points_.n; j++)
	    {
	      if(points_.clusterIndex[j] == points_.clusterIndex[i] and points_.rho[j] > maxrho)
		{
		  maxrho = points_.rho[j];
		  seedIdx = j;
		}
	    }
	  points_.isSeed[seedIdx] = true;
	} 
    }
}

void CLUEAlgo::infoHits()
{
  for(int i = 0; i < points_.n; i++)
    {
      if(points_.clusterIndex[i] == -1) //no outliers
	continue;
      for(int j = 0; j < points_.n; j++)
	{
	  if(points_.clusterIndex[j] == points_.clusterIndex[i])
	    points_.nHitsCluster[i] += 1;
	}
    }
}
  
std::vector<float> CLUEAlgo::getHitsPosX() {
  if(points_.x.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsPosX()" << std::endl;
    throw std::bad_function_call();
  }
  return points_.x;
}

std::vector<float> CLUEAlgo::getHitsPosY() {
  if(points_.y.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsPosY()" << std::endl;
    throw std::bad_function_call();
  }
  return points_.y;
}

std::vector<float> CLUEAlgo::getHitsWeight() {
  if(points_.weight.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsWeight()" << std::endl;
    throw std::bad_function_call();
  }
  return points_.weight;
}

std::vector<int> CLUEAlgo::getHitsClusterId() {
  if(points_.clusterIndex.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsClusterId()" << std::endl;
    throw std::bad_function_call();
  }
  return points_.clusterIndex;
}

std::vector<int> CLUEAlgo::getHitsLayerId() {
  if(points_.layer.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsLayerId()" << std::endl;
    throw std::bad_function_call();
  }
  std::vector<int> layer_output(points_.layer.size());
  for(unsigned int i=0; i<points_.layer.size(); ++i)
    layer_output[i] = points_.layer[i] + 1;
  return layer_output;
}

std::vector<float> CLUEAlgo::getHitsRho() {
  if(points_.rho.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsRho()" << std::endl;
    throw std::bad_function_call();
  }
  return points_.rho;
}

std::vector<float> CLUEAlgo::getHitsDistanceToHighest() {
  if(points_.delta.empty()) {
    std::cout << "ERROR: CLUEAlgo::getDistanceToHighest()" << std::endl;
    throw std::bad_function_call();
  }
  return points_.delta;
}

std::vector<bool> CLUEAlgo::getHitsSeeds() {
  if(points_.isSeed.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsSeeds(): empty" << std::endl;
    throw std::bad_function_call();
  }
  if( std::all_of(points_.isSeed.begin(), points_.isSeed.end(), [](int i) { return i==0; }) ) {
    //this can only happen if all the hits were outliers
    if( ! std::all_of(points_.clusterIndex.begin(), points_.clusterIndex.end(), [](int i) { return i==-1; }) ) 
      {
	std::cout << "ERROR: CLUEAlgo::getHitsSeeds(): all the elements are zero" << std::endl;
	throw std::bad_function_call();
      }
  }
  return points_.isSeed;
}

std::vector<unsigned int> CLUEAlgo::getNHitsInCluster() {
  if(points_.nHitsCluster.empty()) {
    std::cout << "ERROR: CLUEAlgo::getHitsSeeds(): empty" << std::endl;
    throw std::bad_function_call();
  }
  if( std::all_of(points_.nHitsCluster.begin(), points_.nHitsCluster.end(), [](int i) { return i==0; }) ) {
    //this can only happen if all the hits were outliers (see infoHits())
    if( ! std::all_of(points_.clusterIndex.begin(), points_.clusterIndex.end(), [](int i) { return i==-1; }) )
      {
	std::cout << "ERROR: CLUEAlgo::getHitsSeeds(): all the elements are zero. Do not forget to call CLUEAlgo::infoHits()" << std::endl;
	throw std::bad_function_call();
      }
  }
  return points_.nHitsCluster;
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

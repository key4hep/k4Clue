#ifndef CLUEAlgo_h
#define CLUEAlgo_h

// C/C++ headers
#include <set>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <chrono>

#include "LayerTiles.h"
#include "Points.h"

template <typename TILE_CONST>
class CLUEAlgoT {

public:
  CLUEAlgoT(float dc, float rhoc, float outlierDeltaFactor, bool verbose) {
    dc_ = dc; 
    rhoc_ = rhoc;
    outlierDeltaFactor_ = outlierDeltaFactor;
    verbose_ = verbose;
  }
  ~CLUEAlgoT(){} 
    
  // public variables
  float dc_, rhoc_, outlierDeltaFactor_;
  bool verbose_;
    
  Points points_;
  
  bool setPoints(int n, float* x, float* y, int* layer, float* weight) {
    points_.clear();
    // input variables
    for(int i=0; i<n; ++i)
      {
	points_.x.push_back(x[i]);
	points_.y.push_back(y[i]);
	points_.layer.push_back(layer[i]);
	points_.weight.push_back(weight[i]);
      }

    points_.n = points_.x.size();
    if(points_.n == 0)
      return 1;

    // result variables
    points_.rho.resize(points_.n,0);
    points_.delta.resize(points_.n,std::numeric_limits<float>::max());
    points_.nearestHigher.resize(points_.n,-1);
    points_.followers.resize(points_.n);
    points_.clusterIndex.resize(points_.n,-1);
    points_.isSeed.resize(points_.n,0);
    return 0;
  }

  void clearPoints(){ points_.clear(); }

  void makeClusters();
  std::map<int, std::vector<int> > getClusters();
  Points const getPoints() const { return points_; };

  void infoSeeds();
  void infoHits();

  std::string getVerboseString_(unsigned it,
				float x, float y, int layer, float weight,
				float rho, float delta,
				int nh, int isseed, float clusterid,
				unsigned nVerbose) const {
    std::stringstream s;
    std::string sep = ",";
    s << it << sep << x << sep << y << sep;
    s << layer << sep << weight << sep << rho;
    if (delta <= 999)
      s << sep << delta;
    else
      s << ",999"; //convert +inf to 999 in verbose
    s << sep << nh << sep << isseed << sep << clusterid << std::endl;
    return s.str();
  }
  
  void verboseResults(std::string outputFileName="cout", unsigned nVerbose=-1) const {
    if(verbose_)
      {
	if (nVerbose==-1) nVerbose=points_.n;
    
	std::string s;
	s = "index,x,y,layer,weight,rho,delta,nh,isSeed,clusterId\n";
	for(unsigned i=0; i<nVerbose; i++) {
	  s += getVerboseString_(i, points_.x[i], points_.y[i], points_.layer[i],
				 points_.weight[i], points_.rho[i], points_.delta[i],
				 points_.nearestHigher[i], points_.isSeed[i],
				 points_.clusterIndex[i], nVerbose);
	}

	if(outputFileName.compare("cout")==0) //verbose to screen
	  std::cout << s << std::endl;
	else { //verbose to file
	  std::ofstream oFile(outputFileName);
	  oFile << s;
	  oFile.close();
	}
      }
  }
        
private:
  // private member methods
  void prepareDataStructures(std::array<LayerTilesT<TILE_CONST>, TILE_CONST::nLayers> & );
  void calculateLocalDensity(std::array<LayerTilesT<TILE_CONST>, TILE_CONST::nLayers> & );
  void calculateDistanceToHigher(std::array<LayerTilesT<TILE_CONST>, TILE_CONST::nLayers> & );
  void findAndAssignClusters();
  inline float distance(int , int) const ;
};

using CLUEAlgo = CLUEAlgoT<LayerTilesConstants>;
using CLICdetCLUEAlgo = CLUEAlgoT<CLICdetLayerTilesConstants>;
using CLDCLUEAlgo = CLUEAlgoT<CLDLayerTilesConstants>;

#endif

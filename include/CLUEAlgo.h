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

template <typename TILES>
class CLUEAlgo_T {

public:
  CLUEAlgo_T(float dc, float rhoc, float outlierDeltaFactor, bool verbose) {
    dc_ = dc; 
    rhoc_ = rhoc;
    outlierDeltaFactor_ = outlierDeltaFactor;
    verbose_ = verbose;
  }
  ~CLUEAlgo_T(){} 
    
  // public variables
  float dc_, rhoc_, outlierDeltaFactor_;
  bool verbose_;
    
  Points points_;
  
  bool setPoints(int n, float* x, float* y, int* layer, float* weight, float* phi = NULL) {
    points_.clear();
    // input variables
    for(int i=0; i<n; ++i)
      {
	points_.x.push_back(x[i]);
	points_.y.push_back(y[i]);
	points_.layer.push_back(layer[i]);
	points_.weight.push_back(weight[i]);
	if(phi != NULL){
          points_.phi.push_back(phi[i]);
        } else {
          // If the layer tile is declared as endcap, the phi info is not used
          if(TILES::constants_type_t::endcap){
            points_.phi.push_back(0.0);
          } else {
            std::cerr << "ERROR: phi info is not present but you are using a barrel LayerTile! " << std::endl;
          }
        }
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

    // consistency checks
    auto maxLayer = *std::max_element(points_.layer.begin(), points_.layer.end()); 
    if(maxLayer > TILES::constants_type_t::nLayers){
      std::cerr << "Max layer(" << maxLayer << ") is larger "
                << "than the number of layers(" << TILES::constants_type_t::nLayers << ") defined for the current detector" << std::endl;
      return 1;
    }

    auto minX = *std::min_element(points_.x.begin(), points_.x.end()); 
    auto maxX = *std::max_element(points_.x.begin(), points_.x.end()); 
    if(maxX > TILES::constants_type_t::maxX || minX < TILES::constants_type_t::minX){
      std::cout << "Min and/or max x element (" << minX << "," << maxX << ")"
                << " are outside the boundaries defined for the current detector (" << TILES::constants_type_t::minX << "," << TILES::constants_type_t::maxX << ")" << std::endl;
      return 0;
    }

    auto minY = *std::min_element(points_.y.begin(), points_.y.end()); 
    auto maxY = *std::max_element(points_.y.begin(), points_.y.end()); 
    if(maxY > TILES::constants_type_t::maxY || minY < TILES::constants_type_t::minY){
      std::cout << "Min and/or max x element (" << minY << "," << maxY << ")"
                << " are outside the boundaries defined for the current detector (" << TILES::constants_type_t::minY << "," << TILES::constants_type_t::maxY << ")" << std::endl;
      return 0;
    }

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
  void prepareDataStructures(TILES & );
  void calculateLocalDensity(TILES & );
  void calculateDistanceToHigher(TILES & );
  void findAndAssignClusters();
  inline float distance(int i, int j, bool isPhi = false, float r = 0.0) const ;
};

using CLUEAlgo = CLUEAlgo_T<LayerTiles>;
using CLICdetEndcapCLUEAlgo = CLUEAlgo_T<CLICdetEndcapLayerTiles>;
using CLICdetBarrelCLUEAlgo = CLUEAlgo_T<CLICdetBarrelLayerTiles>;
using CLDEndcapCLUEAlgo = CLUEAlgo_T<CLDEndcapLayerTiles>;

#endif

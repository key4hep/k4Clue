#ifndef LayerTiles_h
#define LayerTiles_h

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "LayerTilesConstants.h"
#include "CLICdetEndcapLayerTilesConstants.h"
#include "CLICdetBarrelLayerTilesConstants.h"
#include "CLDEndcapLayerTilesConstants.h"

template <typename T>
class LayerTiles_T {

  public:
    LayerTiles_T(){
      layerTiles_.resize(T::nColumns * T::nRows);
    }

    void fill(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& phi) {
      auto cellsSize = x.size();
      for(unsigned int i = 0; i< cellsSize; ++i) {
        fill(x[i],y[i],phi[i],i);
      }
    }

    void fill(float x, float y, float phi, int i) {
      if(T::endcap){
        layerTiles_[getGlobalBin(x,y)].push_back(i);
      } else { 
        layerTiles_[getGlobalBinPhi(phi,y)].push_back(i);
      }
    }

    int getXBin(float x) const {
      constexpr float xRange = T::maxX - T::minX;
      static_assert(xRange>=0.);
      int xBin = (x - T::minX)*T::rX;
      xBin = std::min(xBin,T::nColumns-1);
      xBin = std::max(xBin,0);
      return xBin;
    }

    int getYBin(float y) const {
      constexpr float yRange = T::maxY - T::minY;
      static_assert(yRange>=0.);
      int yBin = (y - T::minY)*T::rY;
      yBin = std::min(yBin,T::nRows-1);
      yBin = std::max(yBin,0);
      return yBin;
    }

    int getPhiBin(float phi) const {
      auto normPhi = reco::normalizedPhi(phi);
      constexpr float r = T::nColumnsPhi * M_1_PI * 0.5f;
      int phiBin = (normPhi + M_PI) * r;
      return phiBin;
    }

    int getGlobalBin(float x, float y) const {
      return getXBin(x) + getYBin(y)*T::nColumns;
    }

    int getGlobalBinByBin(int xBin, int yBin) const {
      return xBin + yBin*T::nColumns;
    }

    int getGlobalBinPhi(float phi, float y) const {
      return getPhiBin(phi) + getYBin(y)*T::nColumnsPhi;
    }

    int getGlobalBinByBinPhi(int phiBin, int yBin) const {
      return phiBin + yBin*T::nColumnsPhi;
    }

    std::array<int,4> searchBox(float xMin, float xMax, float yMin, float yMax){
      int xBinMin = getXBin(xMin);
      int xBinMax = getXBin(xMax);
      int yBinMin = getYBin(yMin);
      int yBinMax = getYBin(yMax);
      return std::array<int, 4>({{ xBinMin,xBinMax,yBinMin,yBinMax }});
    }

    std::array<int, 4> searchBoxPhiZ(float phiMin, float phiMax, float zMin, float zMax) const {
      int phiBinMin = getPhiBin(phiMin);
      int phiBinMax = getPhiBin(phiMax);
      // If the search window cross the phi-bin boundary, add T::nPhiBins to the
      // MAx value. This guarantees that the caller can perform a valid doule
      // loop on eta and phi. It is the caller responsibility to perform a module
      // operation on the phiBin values returned by this function, to explore the
      // correct bins.
      if (phiBinMax < phiBinMin) {
        phiBinMax += T::nColumnsPhi;
      }
      // In the case of z, I can re-use the Y binning
      int zBinMin = getYBin(zMin);
      int zBinMax = getYBin(zMax);
  
      return std::array<int, 4>({{phiBinMin, phiBinMax, zBinMin, zBinMax}});
    }

    void clear() {
      for(auto& t: layerTiles_) {
        t.clear();
      }
    }


    std::vector<int>& operator[](int globalBinId) {
      return layerTiles_[globalBinId];
    }

  private:
    std::vector< std::vector<int>> layerTiles_;

};

using LayerTiles = LayerTiles_T<LayerTilesConstants>;
using CLICdetEndcapLayerTiles = LayerTiles_T<CLICdetEndcapLayerTilesConstants>;
using CLICdetBarrelLayerTiles = LayerTiles_T<CLICdetBarrelLayerTilesConstants>;
using CLDEndcapLayerTiles = LayerTiles_T<CLDEndcapLayerTilesConstants>;

#endif //LayerTiles_h

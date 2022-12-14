#ifndef LayerTilesGPU_h
#define LayerTilesGPU_h

#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
//GPU Add
#include <cuda_runtime.h>
#include <cuda.h>

#include "GPUVecArray.h"
#include "LayerTilesConstants.h"

template <typename T>
class LayerTilesGPU_T {

  public:

    typedef T type;

    // constructor
    LayerTilesGPU_T(){};

    __device__
    void fill(float x, float y, float phi, int i)
    {
      if(T::endcap){
        layerTiles_[getGlobalBin(x,y)].push_back(i);
      } else { 
        layerTiles_[getGlobalBinPhi(phi,y)].push_back(i);
      }
    }

    __host__ __device__
    int getXBin(float x) const {
      int xBin = (x-T::minX)*T::rX;
      xBin = (xBin<T::nColumns ? xBin:T::nColumns-1);
      xBin = (xBin>0 ? xBin:0);
      return xBin;
    }

    __host__ __device__
    int getYBin(float y) const {
      int yBin = (y-T::minY)*T::rY;
      yBin = (yBin<T::nRows ? yBin:T::nRows-1);
      yBin = (yBin>0 ? yBin:0);;
      return yBin;
    }

    __host__ __device__
    int getPhiBin(float phi) const {
      auto normPhi = reco::normalizedPhi(phi);
      constexpr float r = T::nColumnsPhi * M_1_PI * 0.5f;
      int phiBin = (normPhi + M_PI) * r;
      return phiBin;
    }

    __host__ __device__
    int getGlobalBin(float x, float y) const{
      return getXBin(x) + getYBin(y)*T::nColumns;
    }

    __host__ __device__
    int getGlobalBinPhi(float phi, float y) const {
      return getPhiBin(phi) + getYBin(y)*T::nColumnsPhi;
    }

    __host__ __device__
    int getGlobalBinByBin(int xBin, int yBin) const {
      return xBin + yBin*T::nColumns;
    }

    __host__ __device__
    int4 searchBox(float xMin, float xMax, float yMin, float yMax){
      return int4{ getXBin(xMin), getXBin(xMax), getYBin(yMin), getYBin(yMax)};
    }

    __host__ __device__
    void clear() {
      for(auto& t: layerTiles_) t.reset();
    }

    __host__ __device__
    GPU::VecArray<int, T::maxTileDepth>& operator[](int globalBinId) {
      return layerTiles_[globalBinId];
    }



  private:
    GPU::VecArray<GPU::VecArray<int, T::maxTileDepth>, T::nColumns * T::nRows > layerTiles_;
};

namespace clue {

  using LayerTileGPU = LayerTilesGPU_T<LayerTilesConstants>;
  using TilesGPU = std::array<LayerTileGPU, LayerTilesConstants::nLayers>;

} // end clue namespace

template <typename T>
class GenericTileGPU {
  public:
    // value_type_t is the type of the type of the array used by the incoming <T> type.
    using constants_type_t = typename T::value_type::type;

  private:
    T tiles_;
};

using LayerTilesGPU = GenericTileGPU<clue::TilesGPU>;

#endif

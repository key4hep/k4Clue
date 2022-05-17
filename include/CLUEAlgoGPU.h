#ifndef CLUEAlgoGPU_h
#define CLUEAlgoGPU_h

#include "CLUEAlgo.h"
#include "LayerTilesGPU.h"

static const int maxNSeeds = 100000; 
static const int maxNFollowers = 20; 
static const int localStackSizePerSeed = 20; 

struct PointsPtr {
  float *x; 
  float *y ;
  int *layer ;
  float *weight ;

  float *rho ; 
  float *delta; 
  int *nearestHigher;
  int *clusterIndex; 
  int *isSeed;
};

template<typename TILE_CONST>
class CLUEAlgoGPUT : public CLUEAlgoT<TILE_CONST> {
  // inheriate from CLUEAlgo

  public:
    // constructor
  CLUEAlgoGPUT(float dc, float rhoc, float outlierDeltaFactor, bool verbose) : CLUEAlgo(dc, rhoc, outlierDeltaFactor, verbose) {
      init_device();
    }
    // destructor
    ~CLUEAlgoGPUT(){
      free_device();
    }

    // public methods
    void makeClusters(); // overwrite base class


  private:
    // private variables

    // #ifdef __CUDACC__
    // // CUDA functions

    // algorithm internal variables
    PointsPtr d_points;
    LayerTilesGPUT<TILE_CONST> *d_hist;
    GPU::VecArray<int,maxNSeeds> *d_seeds;
    GPU::VecArray<int,maxNFollowers> *d_followers;

    // private methods
    void init_device(){
      unsigned int reserve = 1000000;
      // input variables
      cudaMalloc(&d_points.x, sizeof(float)*reserve);
      cudaMalloc(&d_points.y, sizeof(float)*reserve);
      cudaMalloc(&d_points.layer, sizeof(int)*reserve);
      cudaMalloc(&d_points.weight, sizeof(float)*reserve);
      // result variables
      cudaMalloc(&d_points.rho, sizeof(float)*reserve);
      cudaMalloc(&d_points.delta, sizeof(float)*reserve);
      cudaMalloc(&d_points.nearestHigher, sizeof(int)*reserve);
      cudaMalloc(&d_points.clusterIndex, sizeof(int)*reserve);
      cudaMalloc(&d_points.isSeed, sizeof(int)*reserve);
      // algorithm internal variables
      cudaMalloc(&d_hist, sizeof(LayerTilesGPUT<TILE_CONST>) * TILE_CONST::nLayers);
      cudaMalloc(&d_seeds, sizeof(GPU::VecArray<int,maxNSeeds>) );
      cudaMalloc(&d_followers, sizeof(GPU::VecArray<int,maxNFollowers>)*reserve);
    }

    void free_device(){
      // input variables
      cudaFree(d_points.x);
      cudaFree(d_points.y);
      cudaFree(d_points.layer);
      cudaFree(d_points.weight);
      // result variables
      cudaFree(d_points.rho);
      cudaFree(d_points.delta);
      cudaFree(d_points.nearestHigher);
      cudaFree(d_points.clusterIndex);
      cudaFree(d_points.isSeed);
      // algorithm internal variables
      cudaFree(d_hist);
      cudaFree(d_seeds);
      cudaFree(d_followers);
    }

    void copy_todevice(){
      // input variables
      cudaMemcpy(d_points.x, CLUEAlgoT<TILE_CONST>::points_.x.data(), sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.y, CLUEAlgoT<TILE_CONST>::points_.y.data(), sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.layer, CLUEAlgoT<TILE_CONST>::points_.layer.data(), sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.weight, CLUEAlgoT<TILE_CONST>::points_.weight.data(), sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyHostToDevice);
    }

    void clear_set(){
      // // result variables
      cudaMemset(d_points.rho, 0x00, sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n);
      cudaMemset(d_points.delta, 0x00, sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n);
      cudaMemset(d_points.nearestHigher, 0x00, sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n);
      cudaMemset(d_points.clusterIndex, 0x00, sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n);
      cudaMemset(d_points.isSeed, 0x00, sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n);
      // algorithm internal variables
      cudaMemset(d_hist, 0x00, sizeof(LayerTilesGPUT<TILE_CONST>) * TILE_CONST::nLayers);
      cudaMemset(d_seeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>));
      cudaMemset(d_followers, 0x00, sizeof(GPU::VecArray<int,maxNFollowers>)*CLUEAlgoT<TILE_CONST>::points_.n);
    }

    void copy_tohost(){
      // result variables
      cudaMemcpy(CLUEAlgoT<TILE_CONST>::points_.clusterIndex.data(), d_points.clusterIndex, sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyDeviceToHost);
      if (CLUEAlgoT<TILE_CONST>::verbose_) {
        // other variables, copy only when verbose_==True
        cudaMemcpy(CLUEAlgoT<TILE_CONST>::points_.rho.data(), d_points.rho, sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(CLUEAlgoT<TILE_CONST>::points_.delta.data(), d_points.delta, sizeof(float)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(CLUEAlgoT<TILE_CONST>::points_.nearestHigher.data(), d_points.nearestHigher, sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(CLUEAlgoT<TILE_CONST>::points_.isSeed.data(), d_points.isSeed, sizeof(int)*CLUEAlgoT<TILE_CONST>::points_.n, cudaMemcpyDeviceToHost);
      }
    }

    // #endif // __CUDACC__
};

using CLUEAlgoGPU = CLUEAlgoGPUT<LayerTilesConstants>;
#endif

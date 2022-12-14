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
  float *phi ;
  int *layer ;
  float *weight ;

  float *rho ; 
  float *delta; 
  int *nearestHigher;
  int *clusterIndex; 
  int *isSeed;
};

template<typename TILE>
class CLUEAlgoGPU_T : public CLUEAlgo_T<TILE> {
  // inheriate from CLUEAlgo

  public:
    // constructor
  CLUEAlgoGPU_T(float dc, float rhoc, float outlierDeltaFactor, bool verbose) : CLUEAlgo(dc, rhoc, outlierDeltaFactor, verbose) {
      init_device();
    }
    // destructor
    ~CLUEAlgoGPU_T(){
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
    LayerTilesGPU_T<TILE> *d_hist;
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
      cudaMalloc(&d_hist, sizeof(LayerTilesGPU_T<TILE>) * TILE::constants_type_t::nLayers);
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
      cudaMemcpy(d_points.x, CLUEAlgo_T<TILE>::points_.x.data(), sizeof(float)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.y, CLUEAlgo_T<TILE>::points_.y.data(), sizeof(float)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.layer, CLUEAlgo_T<TILE>::points_.layer.data(), sizeof(int)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.weight, CLUEAlgo_T<TILE>::points_.weight.data(), sizeof(float)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyHostToDevice);
    }

    void clear_set(){
      // // result variables
      cudaMemset(d_points.rho, 0x00, sizeof(float)*CLUEAlgo_T<TILE>::points_.n);
      cudaMemset(d_points.delta, 0x00, sizeof(float)*CLUEAlgo_T<TILE>::points_.n);
      cudaMemset(d_points.nearestHigher, 0x00, sizeof(int)*CLUEAlgo_T<TILE>::points_.n);
      cudaMemset(d_points.clusterIndex, 0x00, sizeof(int)*CLUEAlgo_T<TILE>::points_.n);
      cudaMemset(d_points.isSeed, 0x00, sizeof(int)*CLUEAlgo_T<TILE>::points_.n);
      // algorithm internal variables
      cudaMemset(d_hist, 0x00, sizeof(LayerTilesGPU_T<TILE>) * TILE::constants_type_t::nLayers);
      cudaMemset(d_seeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>));
      cudaMemset(d_followers, 0x00, sizeof(GPU::VecArray<int,maxNFollowers>)*CLUEAlgo_T<TILE>::points_.n);
    }

    void copy_tohost(){
      // result variables
      cudaMemcpy(CLUEAlgo_T<TILE>::points_.clusterIndex.data(), d_points.clusterIndex, sizeof(int)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyDeviceToHost);
      if (CLUEAlgo_T<TILE>::verbose_) {
        // other variables, copy only when verbose_==True
        cudaMemcpy(CLUEAlgo_T<TILE>::points_.rho.data(), d_points.rho, sizeof(float)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(CLUEAlgo_T<TILE>::points_.delta.data(), d_points.delta, sizeof(float)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(CLUEAlgo_T<TILE>::points_.nearestHigher.data(), d_points.nearestHigher, sizeof(int)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(CLUEAlgo_T<TILE>::points_.isSeed.data(), d_points.isSeed, sizeof(int)*CLUEAlgo_T<TILE>::points_.n, cudaMemcpyDeviceToHost);
      }
    }

    // #endif // __CUDACC__
};

using CLUEAlgoGPU = CLUEAlgoGPU_T<LayerTilesConstants>;
#endif

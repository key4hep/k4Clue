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

class CLUEAlgoGPU : public CLUEAlgo {
  // inheriate from CLUEAlgo

  public:
    // constructor
  CLUEAlgoGPU(float dc, float rhoc, float outlierDeltaFactor, bool verbose) : CLUEAlgo(dc, rhoc, outlierDeltaFactor, verbose) {
      init_device();
    }
    // destructor
    ~CLUEAlgoGPU(){
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
    LayerTilesGPU *d_hist;
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
      cudaMalloc(&d_hist, sizeof(LayerTilesGPU) * NLAYERS);
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
      cudaMemcpy(d_points.x, points_.x.data(), sizeof(float)*points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.y, points_.y.data(), sizeof(float)*points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.layer, points_.layer.data(), sizeof(int)*points_.n, cudaMemcpyHostToDevice);
      cudaMemcpy(d_points.weight, points_.weight.data(), sizeof(float)*points_.n, cudaMemcpyHostToDevice);
    }

    void clear_set(){
      // // result variables
      cudaMemset(d_points.rho, 0x00, sizeof(float)*points_.n);
      cudaMemset(d_points.delta, 0x00, sizeof(float)*points_.n);
      cudaMemset(d_points.nearestHigher, 0x00, sizeof(int)*points_.n);
      cudaMemset(d_points.clusterIndex, 0x00, sizeof(int)*points_.n);
      cudaMemset(d_points.isSeed, 0x00, sizeof(int)*points_.n);
      // algorithm internal variables
      cudaMemset(d_hist, 0x00, sizeof(LayerTilesGPU) * NLAYERS);
      cudaMemset(d_seeds, 0x00, sizeof(GPU::VecArray<int,maxNSeeds>));
      cudaMemset(d_followers, 0x00, sizeof(GPU::VecArray<int,maxNFollowers>)*points_.n);
    }

    void copy_tohost(){
      // result variables
      cudaMemcpy(points_.clusterIndex.data(), d_points.clusterIndex, sizeof(int)*points_.n, cudaMemcpyDeviceToHost);
      if (verbose_) {
        // other variables, copy only when verbose_==True
        cudaMemcpy(points_.rho.data(), d_points.rho, sizeof(float)*points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(points_.delta.data(), d_points.delta, sizeof(float)*points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(points_.nearestHigher.data(), d_points.nearestHigher, sizeof(int)*points_.n, cudaMemcpyDeviceToHost);
        cudaMemcpy(points_.isSeed.data(), d_points.isSeed, sizeof(int)*points_.n, cudaMemcpyDeviceToHost);
      }
    }

    // #endif // __CUDACC__
};

#endif

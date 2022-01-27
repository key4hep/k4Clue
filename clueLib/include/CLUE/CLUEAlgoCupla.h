#ifndef CLUEAlgoCupla_h
#define CLUEAlgoCupla_h

#ifdef FOR_CUDA
#define CUPLA_STREAM_ASYNC_ENABLED 1
#include <cupla/config/GpuCudaRt.hpp>
#elif defined FOR_TBB
#define CUPLA_STREAM_ASYNC_ENABLED 0
#include <cupla/config/CpuTbbBlocks.hpp>
#else
#define CUPLA_STREAM_ASYNC_ENABLED 0
#include <cupla/config/CpuSerial.hpp>
#endif

#include <cuda_to_cupla.hpp>

#include "LayerTilesCupla.h"
#include "CLUEAlgo.h"

// Maximum number of uniques seeds that could be handled. A higher number of
// potential seed will trigger an exception.
static const int maxNSeedsCupla = 8192;

// Maximum number of followers that could be handled. A higher number of
// followers will trigger an exception.
static const int maxNFollowersCupla = 128;

// Maximum size of the local stack used to assign clusters to seeds and
// followers. It should be at least as big as the maximum allowed number of
// followers. Adding more elements with respect to the reserved size will
// trigger an exception.
static const int localStackSizePerSeedCupla = 128;

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

template<typename Acc>
class CLUEAlgoCupla : public CLUEAlgo {

  public:
    CLUEAlgoCupla(float dc, float rhoc, float outlierDeltaFactor, bool verbose)
      : CLUEAlgo(dc, rhoc, outlierDeltaFactor, verbose)
      {
      init_device();
    }
    ~CLUEAlgoCupla(){
      free_device();
    }

    void makeClusters();


  private:

    PointsPtr d_points;
    LayerTilesCupla<Acc> *d_hist;
    GPUCupla::VecArray<int,maxNSeedsCupla> *d_seeds;
    GPUCupla::VecArray<int,maxNFollowersCupla> *d_followers;

    void init_device(){
      unsigned int reserve = 1000000;
      // input variables
      cudaMalloc((void**)&d_points.x, sizeof(float)*reserve);
      cudaMalloc((void**)&d_points.y, sizeof(float)*reserve);
      cudaMalloc((void**)&d_points.layer, sizeof(int)*reserve);
      cudaMalloc((void**)&d_points.weight, sizeof(float)*reserve);
      // result variables
      cudaMalloc((void**)&d_points.rho, sizeof(float)*reserve);
      cudaMalloc((void**)&d_points.delta, sizeof(float)*reserve);
      cudaMalloc((void**)&d_points.nearestHigher, sizeof(int)*reserve);
      cudaMalloc((void**)&d_points.clusterIndex, sizeof(int)*reserve);
      cudaMalloc((void**)&d_points.isSeed, sizeof(int)*reserve);
      // algorithm internal variables
      cudaMalloc((void**)&d_hist, sizeof(LayerTilesCupla<Acc>) * NLAYERS);
      cudaMalloc((void**)&d_seeds, sizeof(GPUCupla::VecArray<int,maxNSeedsCupla>) );
      cudaMalloc((void**)&d_followers, sizeof(GPUCupla::VecArray<int,maxNFollowersCupla>)*reserve);
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
      // result variables
      cudaMemset(d_points.rho, 0x00, sizeof(float)*points_.n);
      cudaMemset(d_points.delta, 0x00, sizeof(float)*points_.n);
      cudaMemset(d_points.nearestHigher, 0x00, sizeof(int)*points_.n);
      cudaMemset(d_points.clusterIndex, 0x00, sizeof(int)*points_.n);
      cudaMemset(d_points.isSeed, 0x00, sizeof(int)*points_.n);
      // algorithm internal variables
      cudaMemset(d_hist, 0x00, sizeof(LayerTilesCupla<Acc>) * NLAYERS);
      cudaMemset(d_seeds, 0x00, sizeof(GPUCupla::VecArray<int,maxNSeedsCupla>));
      cudaMemset(d_followers, 0x00, sizeof(GPUCupla::VecArray<int,maxNFollowersCupla>)*points_.n);
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
};

struct kernel_compute_histogram_opti {
  template <typename T_Acc>
  ALPAKA_FN_ACC
  void operator()(T_Acc const &acc, LayerTilesCupla<T_Acc> *d_hist,
      PointsPtr d_points, int numberOfPoints) const {

    int32_t first = (threadIdx.x + blockIdx.x * blockDim.x) * elemDim.x;
    for(int i = first; i < first + elemDim.x; ++i) {
      if (i < numberOfPoints) {
        // push index of points into tiles
        d_hist[d_points.layer[i]].fill(d_points.x[i], d_points.y[i], i);
      }
    }
  }
};

struct kernel_compute_histogram {
  template <typename T_Acc>
  ALPAKA_FN_ACC
  void operator()(T_Acc const &acc, LayerTilesCupla<T_Acc> *d_hist,
      PointsPtr d_points, int numberOfPoints) const {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numberOfPoints) {
      // push index of points into tiles
      d_hist[d_points.layer[i]].fill(d_points.x[i], d_points.y[i], i);
    }
  }
};

struct kernel_compute_density {
  template <typename T_Acc>
    ALPAKA_FN_ACC
    void operator()(T_Acc const &acc, LayerTilesCupla<T_Acc> *d_hist,
        PointsPtr d_points, float dc,
      int numberOfPoints) const {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numberOfPoints) {
      double rhoi{0.};
      int layeri = d_points.layer[i];
      float xi = d_points.x[i];
      float yi = d_points.y[i];

      // get search box
      int4 search_box =
          d_hist[layeri].searchBox(xi - dc, xi + dc, yi - dc, yi + dc);

      // loop over bins in the search box
      for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
        for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {

          // get the id of this bin
          int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
          // get the size of this bin
          int binSize = d_hist[layeri][binId].size();

          // interate inside this bin
          for (int binIter = 0; binIter < binSize; binIter++) {
            int j = d_hist[layeri][binId][binIter];
            // query N_{dc_}(i)
            float xj = d_points.x[j];
            float yj = d_points.y[j];
            float dist_ij =
                std::sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
            if (dist_ij <= dc) {
              // sum weights within N_{dc_}(i)
              rhoi += (i == j ? 1.f : 0.5f) * d_points.weight[j];
            }
          } // end of interate inside this bin
        }
      } // end of loop over bins in search box
      d_points.rho[i] = rhoi;
    }
  }
};

struct kernel_compute_distanceToHigher {
  template <typename T_Acc>
  ALPAKA_FN_ACC
  void operator()(T_Acc const &acc, LayerTilesCupla<T_Acc> *d_hist,
		  PointsPtr d_points,
		  float outlierDeltaFactor,
		  float dc,
		  int numberOfPoints) const {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float dm = outlierDeltaFactor * dc;
    
    if (i < numberOfPoints) {
      int layeri = d_points.layer[i];

      float deltai = std::numeric_limits<float>::max();
      int nearestHigheri = -1;
      float xi = d_points.x[i];
      float yi = d_points.y[i];
      float rhoi = d_points.rho[i];

      // get search box
      int4 search_box =
          d_hist[layeri].searchBox(xi - dm, xi + dm, yi - dm, yi + dm);

      // loop over all bins in the search box
      for (int xBin = search_box.x; xBin < search_box.y + 1; ++xBin) {
        for (int yBin = search_box.z; yBin < search_box.w + 1; ++yBin) {
          // get the id of this bin
          int binId = d_hist[layeri].getGlobalBinByBin(xBin, yBin);
          // get the size of this bin
          int binSize = d_hist[layeri][binId].size();

          // interate inside this bin
          for (int binIter = 0; binIter < binSize; binIter++) {
            int j = d_hist[layeri][binId][binIter];
            // query N'_{dm}(i)
            float xj = d_points.x[j];
            float yj = d_points.y[j];
            float dist_ij =
                std::sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj));
            bool foundHigher = (d_points.rho[j] > rhoi);
            // in the rare case where rho is the same, use detid
            foundHigher = foundHigher || ((d_points.rho[j] == rhoi) && (j > i));
            if (foundHigher && dist_ij <= dm) { // definition of N'_{dm}(i)
              // find the nearest point within N'_{dm}(i)
              if (dist_ij < deltai) {
                // update deltai and nearestHigheri
                deltai = dist_ij;
                nearestHigheri = j;
              }
            }
          } // end of interate inside this bin
        }
      } // end of loop over bins in search box
      d_points.delta[i] = deltai;
      d_points.nearestHigher[i] = nearestHigheri;
    }
  }
};

struct kernel_find_clusters {
  template <typename T_Acc>
  ALPAKA_FN_ACC
  void operator()(T_Acc const &acc, GPUCupla::VecArray<int, maxNSeedsCupla> *d_seeds,
		  GPUCupla::VecArray<int, maxNFollowersCupla> *d_followers, PointsPtr d_points,
		  float outlierDeltaFactor, float dc, float rhoc, int numberOfPoints) const {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numberOfPoints) {
      // initialize clusterIndex
      d_points.clusterIndex[i] = -1;
      // determine seed or outlier
      float deltai = d_points.delta[i];
      float rhoi = d_points.rho[i];
      bool isSeed = (deltai > dc) && (rhoi >= rhoc);
      bool isOutlier = (deltai > outlierDeltaFactor * dc) && (rhoi < rhoc);

      if (isSeed) {
        // set isSeed as 1
        d_points.isSeed[i] = 1;
        d_seeds[0].push_back(acc, i); // head of d_seeds
      } else {
        if (!isOutlier) {
          assert(d_points.nearestHigher[i] < numberOfPoints);
          // register as follower at its nearest higher
          d_followers[d_points.nearestHigher[i]].push_back(acc, i);
        }
      }
    }
  }
};

struct kernel_assign_clusters {
  template <typename T_Acc>
  ALPAKA_FN_ACC
  void operator()(T_Acc const &acc,
      GPUCupla::VecArray<int, maxNSeedsCupla> *d_seeds,
      GPUCupla::VecArray<int, maxNFollowersCupla> *d_followers,
      PointsPtr d_points) const {

    int idxCls = blockIdx.x * blockDim.x + threadIdx.x;

    if (idxCls < d_seeds[0].size()) {

      int localStack[localStackSizePerSeedCupla] = {-1};
      int localStackSize = 0;

      // assign cluster to seed[idxCls]
      int idxThisSeed = d_seeds[0][idxCls];
      d_points.clusterIndex[idxThisSeed] = idxCls;
      // push_back idThisSeed to localStack
      assert(("Local stack size too small", localStackSize < localStackSizePerSeedCupla));
      localStack[localStackSize] = idxThisSeed;
      localStackSize++;

      // process all elements in localStack
      while (localStackSize > 0) {
        // get last element of localStack
        assert(("Local stack size too small", localStackSize-1 < localStackSizePerSeedCupla));
        int idxEndOflocalStack = localStack[localStackSize - 1];

        int temp_clusterIndex = d_points.clusterIndex[idxEndOflocalStack];
        // pop_back last element of localStack
        assert(("Local stack size too small", localStackSize-1 < localStackSizePerSeedCupla));
        localStack[localStackSize - 1] = -1;
        localStackSize--;

        // loop over followers of last element of localStack
        for (int j : d_followers[idxEndOflocalStack]) {
          // pass id to follower
          d_points.clusterIndex[j] = temp_clusterIndex;
          // push_back follower to localStack
          assert(("Local stack size too small", localStackSize < localStackSizePerSeedCupla));
          localStack[localStackSize] = j;
          localStackSize++;
        }
      }
    }
  }
};

template<typename Acc>
void CLUEAlgoCupla<Acc>::makeClusters() {

  copy_todevice();
  clear_set();

  ////////////////////////////////////////////
  // calculate rho, delta and find seeds
  // 1 point per thread
  ////////////////////////////////////////////
#ifdef FOR_CUDA
  const dim3 blockSize(1024, 1, 1);
  const dim3 blockSize_opti(1024, 1, 1);
#endif

#ifdef FOR_TBB
  const dim3 blockSize(1, 1, 1);
  const dim3 blockSize_opti(4096, 1, 1);
#endif

  const dim3 gridSize(ceil(points_.n/ (float)blockSize.x), 1, 1);
  const dim3 gridSize_opti(ceil(points_.n/ (float)blockSize_opti.x), 1, 1);

#ifdef FOR_CUDA
  auto start = std::chrono::high_resolution_clock::now();
  CUPLA_KERNEL(kernel_compute_histogram)(gridSize, blockSize, 0, 0)(d_hist, d_points, points_.n);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "--- prepareDataStructures:     " << elapsed.count() *1000 << " ms\n";
#endif

#ifdef FOR_TBB
  auto start = std::chrono::high_resolution_clock::now();
  CUPLA_KERNEL_OPTI(kernel_compute_histogram_opti)(gridSize_opti, blockSize_opti, 0, 0)(d_hist, d_points, points_.n);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "--- prepareDataStructures_opti:     " << elapsed.count() *1000 << " ms\n";
#endif


  start = std::chrono::high_resolution_clock::now();
  CUPLA_KERNEL(kernel_compute_density)(gridSize, blockSize, 0, 0)(d_hist, d_points, dc_, points_.n);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- calculateDistanceToHigher: " << elapsed.count() *1000 << " ms\n";


  start = std::chrono::high_resolution_clock::now();
  CUPLA_KERNEL(kernel_compute_distanceToHigher)(gridSize, blockSize, 0, 0)(d_hist, d_points, outlierDeltaFactor_, dc_, points_.n);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- calculateLocalDensity:     " << elapsed.count() *1000 << " ms\n";

  start = std::chrono::high_resolution_clock::now();
  CUPLA_KERNEL(kernel_find_clusters)(gridSize, blockSize, 0, 0)(d_seeds, d_followers, d_points, outlierDeltaFactor_, dc_, rhoc_, points_.n);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- findSeedAndFollowers:      " << elapsed.count() *1000 << " ms\n";

  ////////////////////////////////////////////
  // assign clusters
  // 1 point per seeds
  ////////////////////////////////////////////
  start = std::chrono::high_resolution_clock::now();
  const dim3 gridSize_nseeds(ceil(maxNSeedsCupla / (float)blockSize.x), 1, 1);
  CUPLA_KERNEL(kernel_assign_clusters)(gridSize_nseeds, blockSize, 0, 0)(d_seeds, d_followers, d_points);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "--- assignClusters:            " << elapsed.count() *1000 << " ms" << std::endl;

  copy_tohost();
}
#endif

/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CLUEAlgo_h
#define CLUEAlgo_h

// C/C++ headers
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "LayerTiles.h"
#include "Points.h"

template <typename TILES>
class CLUEAlgo_T {

public:
  CLUEAlgo_T() {
    dc_ = 0.0;
    rhoc_ = 0.0;
    outlierDeltaFactor_ = 0.0;
    verbose_ = false;
  }
  CLUEAlgo_T(float dc, float rhoc, float outlierDeltaFactor, bool verbose) {
    dc_ = dc;
    rhoc_ = rhoc;
    outlierDeltaFactor_ = outlierDeltaFactor;
    verbose_ = verbose;
    if (verbose_) {
      std::cout << "ClueGaudiAlgorithmWrapper: nTiles (cols,rows):     " << TILES::constants_type_t::nTiles;
      std::cout << " ("
                << (TILES::constants_type_t::endcap ? TILES::constants_type_t::nColumns
                                                    : TILES::constants_type_t::nColumnsPhi);
      std::cout << "," << TILES::constants_type_t::nRows << " )" << std::endl;
    }
  }

  // public variables
  float dc_, rhoc_, outlierDeltaFactor_;
  bool verbose_;

  Points points_;

  bool clearAndSetPoints(int n, float* x, float* y, int* layer, float* weight, float* r = NULL) {
    points_.clear();
    // input variables
    for (int i = 0; i < n; ++i) {
      points_.x.push_back(x[i]);
      points_.y.push_back(y[i]);
      points_.layer.push_back(layer[i]);
      points_.weight.push_back(weight[i]);
      if (r != NULL) {
        points_.r.push_back(r[i]);
      } else {
        // If the layer tile is declared as endcap, the r info is not used
        if (TILES::constants_type_t::endcap) {
          points_.r.push_back(0.0);
        } else {
          std::cerr << "ERROR: r info is not present but you are using a barrel LayerTile! " << std::endl;
        }
      }
    }

    points_.n = points_.x.size();
    if (points_.n == 0)
      return 1;

    // result variables
    points_.rho.resize(points_.n, 0);
    points_.delta.resize(points_.n, std::numeric_limits<float>::max());
    points_.nearestHigher.resize(points_.n, -1);
    points_.followers.resize(points_.n);
    points_.clusterIndex.resize(points_.n, -1);
    points_.isSeed.resize(points_.n, 0);

    // consistency checks
    auto maxLayer = *std::max_element(points_.layer.begin(), points_.layer.end());
    if (maxLayer > TILES::constants_type_t::nLayers) {
      std::cerr << "Max layer(" << maxLayer << ") is larger "
                << "than the number of layers(" << TILES::constants_type_t::nLayers
                << ") defined for the current detector" << std::endl;
      return 1;
    }

    auto minX = *std::min_element(points_.x.begin(), points_.x.end());
    auto maxX = *std::max_element(points_.x.begin(), points_.x.end());
    if (maxX > TILES::constants_type_t::maxX || minX < TILES::constants_type_t::minX) {
      std::cout << "Min and/or max x element (" << minX << "," << maxX << ")"
                << " are outside the boundaries defined for the current detector (" << TILES::constants_type_t::minX
                << "," << TILES::constants_type_t::maxX << ")" << std::endl;
      return 0;
    }

    auto minY = *std::min_element(points_.y.begin(), points_.y.end());
    auto maxY = *std::max_element(points_.y.begin(), points_.y.end());
    if (maxY > TILES::constants_type_t::maxY || minY < TILES::constants_type_t::minY) {
      std::cout << "Min and/or max x element (" << minY << "," << maxY << ")"
                << " are outside the boundaries defined for the current detector (" << TILES::constants_type_t::minY
                << "," << TILES::constants_type_t::maxY << ")" << std::endl;
      return 0;
    }

    return 0;
  }

  void clearPoints() { points_.clear(); }
  void clearLayerTiles() {
    for (unsigned i = 0; i < TILES::constants_type_t::nLayers; i++) {
      allLayerTiles_[i].clear();
    }
  }
  void makeClusters();
  std::map<int, std::vector<int>> getClusters();
  Points const getPoints() const { return points_; };

  void infoSeeds();
  void infoHits();

  std::string getVerboseString_(unsigned it, float x, float y, int layer, float weight, float rho, float delta, int nh,
                                int isseed, float clusterid) const {
    std::stringstream s;
    std::string sep = ",";
    s << it << sep << x << sep << y << sep;
    s << layer << sep << weight << sep << rho;
    if (delta <= 999)
      s << sep << delta;
    else
      s << ",999"; // convert +inf to 999 in verbose
    s << sep << nh << sep << isseed << sep << clusterid << std::endl;
    return s.str();
  }

  void verboseResults(std::string outputFileName = "cout", int nVerbose = -1) const {
    if (verbose_) {
      if (nVerbose == -1)
        nVerbose = points_.n;

      std::string s;
      s = "index,x,y,layer,weight,rho,delta,nh,isSeed,clusterId\n";
      for (int i = 0; i < nVerbose; i++) {
        s += getVerboseString_(i, points_.x[i], points_.y[i], points_.layer[i], points_.weight[i], points_.rho[i],
                               points_.delta[i], points_.nearestHigher[i], points_.isSeed[i], points_.clusterIndex[i]);
      }

      if (outputFileName.compare("cout") == 0) // verbose to screen
        std::cout << s << std::endl;
      else { // verbose to file
        std::ofstream oFile(outputFileName);
        oFile << s;
        oFile.close();
      }
    }
  }

private:
  // private member methods
  void prepareDataStructures();
  void calculateLocalDensity();
  void calculateDistanceToHigher();
  void findAndAssignClusters();
  inline float distance(int i, int j, bool isPhi = false, float r = 0.0) const;
  inline float distance2(int i, int j, bool isPhi = false, float r = 0.0) const;
  TILES allLayerTiles_;
};

using CLUEAlgo = CLUEAlgo_T<LayerTiles>;
using CLICdetEndcapCLUEAlgo = CLUEAlgo_T<CLICdetEndcapLayerTiles>;
using CLICdetBarrelCLUEAlgo = CLUEAlgo_T<CLICdetBarrelLayerTiles>;
using CLDEndcapCLUEAlgo = CLUEAlgo_T<CLDEndcapLayerTiles>;
using CLDBarrelCLUEAlgo = CLUEAlgo_T<CLDBarrelLayerTiles>;
using LArBarrelCLUEAlgo = CLUEAlgo_T<LArBarrelLayerTiles>;

#endif

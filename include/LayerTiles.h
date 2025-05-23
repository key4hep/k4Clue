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
#ifndef LayerTiles_h
#define LayerTiles_h

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "CLDBarrelLayerTilesConstants.h"
#include "CLDEndcapLayerTilesConstants.h"
#include "CLICdetBarrelLayerTilesConstants.h"
#include "CLICdetEndcapLayerTilesConstants.h"
#include "LArBarrelLayerTilesConstants.h"
#include "LayerTilesConstants.h"

template <typename T>
class LayerTiles_T {

public:
  typedef T type;

  LayerTiles_T() { layerTiles_.resize(T::nTiles); }

  void fill(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& phi) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      fill(x[i], y[i], phi[i], i);
    }
  }

  void fill(float x, float y, float phi, int i) {
    if (T::endcap) {
      layerTiles_[getGlobalBin(x, y)].push_back(i);
    } else {
      layerTiles_[getGlobalBinPhi(phi, y)].push_back(i);
    }
  }

  int getXBin(float x) const {
    constexpr float xRange = T::maxX - T::minX;
    static_assert(xRange >= 0.);
    int xBin = (x - T::minX) * T::rX;
    xBin = std::min(xBin, T::nColumns - 1);
    xBin = std::max(xBin, 0);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = T::maxY - T::minY;
    static_assert(yRange >= 0.);
    int yBin = (y - T::minY) * T::rY;
    yBin = std::min(yBin, T::nRows - 1);
    yBin = std::max(yBin, 0);
    return yBin;
  }

  int getPhiBin(float phi) const {
    auto normPhi = reco::normalizedPhi(phi);
    constexpr float r = T::nColumnsPhi * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;
    return phiBin;
  }

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * T::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * T::nColumns; }

  int getGlobalBinPhi(float phi, float y) const { return getPhiBin(phi) + getYBin(y) * T::nColumnsPhi; }

  int getGlobalBinByBinPhi(int phiBin, int yBin) const { return phiBin + yBin * T::nColumnsPhi; }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) const {
    int xBinMin = getXBin(xMin);
    int xBinMax = getXBin(xMax);
    int yBinMin = getYBin(yMin);
    int yBinMax = getYBin(yMax);
    return std::array<int, 4>({{xBinMin, xBinMax, yBinMin, yBinMax}});
  }

  /**
   * If the search window cross the phi-bin boundary, add T::nPhiBins to the
   * max value. This guarantees that the caller can perform a valid double
   * loop on eta and phi. It is the caller responsibility to perform a modulo
   * operation on the phiBin values returned by this function, to explore the
   * correct bins.
   */
  std::array<int, 4> searchBoxPhiZ(float phiMin, float phiMax, float zMin, float zMax) const {
    int phiBinMin = getPhiBin(phiMin);
    int phiBinMax = getPhiBin(phiMax);
    if (phiBinMax < phiBinMin) {
      phiBinMax += T::nColumnsPhi;
    }
    // In the case of z, I can re-use the Y binning
    int zBinMin = getYBin(zMin);
    int zBinMax = getYBin(zMax);

    return std::array<int, 4>({{phiBinMin, phiBinMax, zBinMin, zBinMax}});
  }

  void clear() {
    for (auto& t : layerTiles_) {
      t.clear();
    }
  }

  std::vector<int>& operator[](int globalBinId) { return layerTiles_[globalBinId]; }

  const std::vector<int>& operator[](int globalBinId) const { return layerTiles_[globalBinId]; }

private:
  std::vector<std::vector<int>> layerTiles_;
};

namespace clue {

using LayerTile = LayerTiles_T<LayerTilesConstants>;
using Tiles = std::array<LayerTile, LayerTilesConstants::nLayers>;

using CLICdetEndcapLayerTile = LayerTiles_T<CLICdetEndcapLayerTilesConstants>;
using CLICdetEndcapTiles = std::array<CLICdetEndcapLayerTile, CLICdetEndcapLayerTilesConstants::nLayers>;

using CLICdetBarrelLayerTile = LayerTiles_T<CLICdetBarrelLayerTilesConstants>;
using CLICdetBarrelTiles = std::array<CLICdetBarrelLayerTile, CLICdetBarrelLayerTilesConstants::nLayers>;

using CLDEndcapLayerTile = LayerTiles_T<CLDEndcapLayerTilesConstants>;
using CLDEndcapTiles = std::array<CLDEndcapLayerTile, CLDEndcapLayerTilesConstants::nLayers>;

using CLDBarrelLayerTile = LayerTiles_T<CLDBarrelLayerTilesConstants>;
using CLDBarrelTiles = std::array<CLDBarrelLayerTile, CLDBarrelLayerTilesConstants::nLayers>;

using LArBarrelLayerTile = LayerTiles_T<LArBarrelLayerTilesConstants>;
using LArBarrelTiles = std::array<LArBarrelLayerTile, LArBarrelLayerTilesConstants::nLayers>;

} // namespace clue

template <typename T>
class GenericTile {
public:
  // value_type_t is the type of the type of the array used by the incoming <T> type.
  using constants_type_t = typename T::value_type::type;
  // This class represents a generic collection of Tiles. The additional index
  // numbering is not handled internally. It is the user's responsibility to
  // properly use and consistently access it here.
  const auto& operator[](int index) const { return tiles_[index]; }
  auto& operator[](int index) { return tiles_[index]; }
  void fill(int index, float x, float y, float phi, unsigned int objectId) { tiles_[index].fill(x, y, phi, objectId); }

private:
  T tiles_;
};

using LayerTiles = GenericTile<clue::Tiles>;
using CLICdetEndcapLayerTiles = GenericTile<clue::CLICdetEndcapTiles>;
using CLICdetBarrelLayerTiles = GenericTile<clue::CLICdetBarrelTiles>;
using CLDEndcapLayerTiles = GenericTile<clue::CLDEndcapTiles>;
using CLDBarrelLayerTiles = GenericTile<clue::CLDBarrelTiles>;
using LArBarrelLayerTiles = GenericTile<clue::LArBarrelTiles>;

#endif // LayerTiles_h

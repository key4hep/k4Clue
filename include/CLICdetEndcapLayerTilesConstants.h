/*
 * Copyright (c) 2020-2023 Key4hep-Project.
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
#ifndef CLICdetEndcapLayerTilesConstants_h
#define CLICdetEndcapLayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct CLICdetEndcapLayerTilesConstants {

  // Global x coordinate
  static constexpr float minX =  -1701.f;
  static constexpr float maxX =   1701.f;

  // Global y coordinate
  static constexpr float minY =  -1701.f;
  static constexpr float maxY =   1701.f;

  static constexpr float tileSize = 30.f;
  static constexpr float tileSizePhi = 0.10f;
  static constexpr int nColumns = reco::ceil((maxX-minX)/tileSize);
  static constexpr int nColumnsPhi = reco::ceil(2. * M_PI / tileSizePhi);
  static constexpr int nRows    = reco::ceil((maxY-minY)/tileSize);
  static constexpr int maxTileDepth = 40;

  static constexpr float rX = nColumns/(maxX-minX);
  static constexpr float rY = nRows/(maxY-minY);

  static constexpr int nTiles = nColumns * nRows;

  static constexpr int nLayers = 80; // Includes EE+ and EE-
  static constexpr bool endcap = true;
};

#endif // CLICdetEndcapLayerTilesConstants_h

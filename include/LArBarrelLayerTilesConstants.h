/**
 * Variables meaning is specified in
 * include/readme.md
*/

#ifndef LArBarrelLayerTilesConstants_h
#define LArBarrelLayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct LArBarrelLayerTilesConstants {

  static constexpr float minX =  -8700.f;
  static constexpr float maxX =   8700.f;
  static constexpr float minY =  -3110.f;
  static constexpr float maxY =   3110.f;
  static constexpr float tileSize = 50.f;
  static constexpr float tileSizePhi = 0.15f;
  static constexpr int nColumns = reco::ceil((maxX-minX)/tileSize);
  static constexpr int nColumnsPhi = reco::ceil(2. * M_PI / tileSizePhi);
  static constexpr int nRows    = reco::ceil((maxY-minY)/tileSize);
  static constexpr int maxTileDepth = 40;

  static constexpr float rX = nColumns/(maxX-minX);
  static constexpr float rY = nRows/(maxY-minY);

  static constexpr int nTiles = nColumnsPhi * nRows;

  static constexpr int nLayers = 12;
  static constexpr bool endcap = false;
};

#endif // LArBarrelLayerTilesConstants_h



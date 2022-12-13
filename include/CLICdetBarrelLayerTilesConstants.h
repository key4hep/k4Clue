#ifndef CLICdetBarrelLayerTilesConstants_h
#define CLICdetBarrelLayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct CLICdetBarrelLayerTilesConstants {

  // Global r*phi coordinate
  static constexpr float minX =  -6000.f;
  static constexpr float maxX =   6000.f;

  // Global z coordinate
  static constexpr float minY =  -2210.f;
  static constexpr float maxY =   2210.f;

  static constexpr float tileSize = 15.f;
  static constexpr float tileSizePhi = 0.01f;
  static constexpr int nColumns = reco::ceil((maxX-minX)/tileSize);
  static constexpr int nColumnsPhi = reco::ceil(2. * M_PI / tileSizePhi);
  static constexpr int nRows    = reco::ceil((maxY-minY)/tileSize);
  static constexpr int maxTileDepth = 40;

  static constexpr float rX = nColumns/(maxX-minX);
  static constexpr float rY = nRows/(maxY-minY);

  static constexpr int nTiles = nColumnsPhi * nRows;

  static constexpr int nLayers = 40;
  static constexpr bool endcap = false;
};

#endif // CLICdetBarrelLayerTilesConstants_h

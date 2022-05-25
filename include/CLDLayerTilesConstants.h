#ifndef CLDLayerTilesConstants_h
#define CLDLayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct CLDLayerTilesConstants {

  static constexpr float minX =  -2455.f;
  static constexpr float maxX =   2455.f;
  static constexpr float minY =  -2455.f;
  static constexpr float maxY =   2455.f;
  static constexpr float tileSize = 15.f;
  static constexpr int nColumns = reco::ceil((maxX-minX)/tileSize);
  static constexpr int nRows    = reco::ceil((maxY-minY)/tileSize);
  static constexpr int maxTileDepth = 40;

  static constexpr float rX = nColumns/(maxX-minX);
  static constexpr float rY = nRows/(maxY-minY);

  static constexpr int nTiles = nColumns * nRows;

  static constexpr int nLayers = 40;
};

#endif // CLDLayerTilesConstants_h

#ifndef CLICdetLayerTilesConstants_h
#define CLICdetLayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct CLICdetLayerTilesConstants {

  static constexpr float minX =  -230.f;
  static constexpr float maxX =   230.f;
  static constexpr float minY =  -250.f;
  static constexpr float maxY =   250.f;
  static constexpr float tileSize = .5f;
  static constexpr int nColumns = reco::ceil((maxX-minX)/tileSize);
  static constexpr int nRows    = reco::ceil((maxY-minY)/tileSize);
  static constexpr int maxTileDepth = 40;

  static constexpr float rX = nColumns/(maxX-minX);
  static constexpr float rY = nRows/(maxY-minY);

  static constexpr int nTiles = nColumns * nRows;

  static constexpr int nLayers = 40;
};

#endif // CLICdetLayerTilesConstants_h

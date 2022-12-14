#ifndef LayerTilesConstants_h
#define LayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct LayerTilesConstants {

  static constexpr float minX =  -250.f;
  static constexpr float maxX =   250.f;
  static constexpr float minY =  -250.f;
  static constexpr float maxY =   250.f;
  static constexpr float tileSize = 5.f;
  static constexpr float tileSizePhi = 0.15f;
  static constexpr int nColumns = reco::ceil((maxX-minX)/tileSize);
  static constexpr int nColumnsPhi = reco::ceil(2. * M_PI / tileSizePhi);
  static constexpr int nRows    = reco::ceil((maxY-minY)/tileSize);
  static constexpr int maxTileDepth = 40;

  static constexpr float rX = nColumns/(maxX-minX);
  static constexpr float rY = nRows/(maxY-minY);

  static constexpr int nTiles = nColumns * nRows;

  static constexpr int nLayers = 100;
  static constexpr bool endcap = true;
};

#endif // LayerTilesConstants_h

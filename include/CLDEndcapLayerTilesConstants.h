#ifndef CLDEndcapLayerTilesConstants_h
#define CLDEndcapLayerTilesConstants_h

#include "constexpr_cmath.h"
#include <array>

struct CLDEndcapLayerTilesConstants {

  static constexpr float minX =  -2455.f;
  static constexpr float maxX =   2455.f;
  static constexpr float minY =  -2455.f;
  static constexpr float maxY =   2455.f;
  static constexpr float tileSize = 15.f;
  static constexpr float tileSizePhi = 0.01f;
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

#endif // CLDEndcapLayerTilesConstants_h

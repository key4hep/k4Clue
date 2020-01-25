#ifndef LayerTilesConstants_h
#define LayerTilesConstants_h

#include <cstdint>
#include <array>


#define NLAYERS 100

namespace LayerTilesConstants {

  constexpr int32_t ceil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num) ? static_cast<int32_t>(num) : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
  }

  constexpr float minX =  -250.f;
  constexpr float maxX =   250.f;
  constexpr float minY =  -250.f;
  constexpr float maxY =   250.f;
  constexpr float tileSize = 5.f;
  constexpr int nColumns = LayerTilesConstants::ceil((maxX-minX)/tileSize);
  constexpr int nRows    = LayerTilesConstants::ceil((maxY-minY)/tileSize);
  constexpr int maxTileDepth = 40;

  constexpr float rX = nColumns/(maxX-minX);
  constexpr float rY = nRows/(maxY-minY);

}

#endif // LayerTilesConstants_h

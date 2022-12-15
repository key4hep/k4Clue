# A step-by-step guide to introduce a new detector

The `LayerTiles` created for your new detector must be templated with 
the corresponding `\*LayerTilesConstants.h`.

## 1. Look at the existing example

All variable listed in the [LayerTilesConstants.h](LayerTilesConstants.h) must be present and specified:
* [min, max][X, Y]
* tileSize[-, Phi]
* nColumns[-, Phi]
* nRows
* maxTileDepth
* rX, rY, nTiles (formula to be kept)
* nLayers
* endcap (bool)

In the case of an endcap layer, only the variables `tileSize` and `nColumns` are used and the coordinates ( $\textnormal{x}, \textnormal{y}$ ) 
corresponds to the transverse plane positions.

In the case of a barrel layer, only the variables `tileSizePhi` and `nColumnsPhi` are used and the nTiles must be defined as:
```c++
static constexpr int nTiles = nColumnsPhi * nRows;
```
The coordinates ( $\textnormal{x}, \textnormal{y}$ ) correspond to ( $\textnormal{r} \cdot \phi, \textnormal{z}$ ), respectively.

For more details regarding the barrel extension, have a look at [these slides](https://indico.cern.ch/event/1207900/#3-k4clue-update)
of Oct 2022.
 
## 2. Changes in the templated classes 

After having created your new detector layer tiles, for example in `MyDetLayerTilesConstants.h`,
you need to make the `CLUE` classes aware of this.

Include the new header in [LayerTiles.h](LayerTiles.h):
```c++
#include "MyDetLayerTilesConstants.h"
```
and create the layer tiles with the new constants at the end of the file:
```c++
namespace clue {

  ...

  using MyDetLayerTile = LayerTiles_T<MyDetLayerTilesConstants>;
  using MyDetTiles = std::array<MyDetLayerTile, MyDetLayerTilesConstants::nLayers>;

} // end clue namespace

...

using MyDetLayerTiles = GenericTile<clue::MyDetTiles>;
```

Include almost at the end of [CLUEAlgo.h](CLUEAlgo.h) the following line:
```c++
using MyDetCLUEAlgo = CLUEAlgo_T<MyDetLayerTiles>;
```

Explicit template instantiation at the end of [CLUEAlgo.cc](../src/CLUEAlgo.cc):
```c++
template class CLUEAlgo_T<MyDetLayerTiles>;
```

If you want to test it also on the GPU verison of CLUE, 
similar changes must be included also in [CLUEAlgoGPU.h](CLUEAlgoGPU.h) and [LayerTilesGPU.h](LayerTilesGPU.h).
 
## 3. Use it! 

You can use the templated version of CLUE with the new tiles built on your detector with
```
    CLUEAlgo_T<MyDetLayerTilesConstants> clueAlgo(dc, rhoc, outlierDeltaFactor, false);
```
or
```
    MyDetCLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, false);
```

Of course, the last step is to recompile the code in the main repository:
```bash
cd build/
make
```




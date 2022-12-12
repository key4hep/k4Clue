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

In the case of an endcap layer, only the variables `tileSize` and `nColumns` are used and the coordinates (x, y) 
corresponds to the transverse plane positions.

In the case of a barrel layer, the coordinates (x, y) correspond to (r$\phi$, z), respectively.

For more details regarding the barrel extension, have a look at [these slides](https://indico.cern.ch/event/1207900/#3-k4clue-update)
of Oct 2022.
 
## 2. Changes in the templated classes 

After having created your new detector layer tiles, for example in `MyDetLayerTilesConstants.h`,
you need to make the `CLUE` classes aware of this.

Include the new header in [LayerTiles.h](LayerTiles.h):
```c++
#include "MyDetLayerTilesConstants.h"
```

Include almost at the end of [CLUEAlgo.h](CLUEAlgo.h) the following line:
```c++
using MyDetCLUEAlgo = CLUEAlgo_T<MyDetLayerTilesConstants>;
```

Explicit template instantiation at the end of [CLUEAlgo.cc](../src/CLUEAlgo.cc):
```c++
template class CLUEAlgo_T<MyDetLayerTilesConstants>;
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




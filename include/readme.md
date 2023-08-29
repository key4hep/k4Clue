<!--
Copyright (c) 2020-2023 Key4hep-Project.

This file is part of Key4hep.
See https://key4hep.github.io/key4hep-doc/ for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
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

In the case of an endcap layer, the coordinates ( $\textnormal{x}, \textnormal{y}$ ) 
corresponds to the transverse plane positions and only the variables `tileSize` and `nColumns` are used.

In the case of a barrel layer, the coordinates ( $\textnormal{x}, \textnormal{y}$ ) correspond to ( $\textnormal{r} \cdot \phi, \textnormal{z}$ ), respectively.
Moreover, only the variables `tileSizePhi` and `nColumnsPhi` are used and the nTiles must be defined as:
```c++
static constexpr int nTiles = nColumnsPhi * nRows;
```

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




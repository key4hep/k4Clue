[![linux](https://github.com/key4hep/k4Clue/actions/workflows/test.yml/badge.svg)](https://github.com/key4hep/k4Clue/actions/workflows/test.yml)

![Logo](plots/k4Clue_logo.png)

# k4CLUE: The CLUE algorithm for future colliders (on CPU and GPU)

## Table of contents
* [General info](#general-info)
* [Setup the environment](#setup-the-environment)
* [Setup CLUE](#setup-clue)
* [Examples of use](#examples-of-use)
* [Package maintainer](#package-maintainer)

## General info

The CLUE algorithm ([here](https://gitlab.cern.ch/kalos/clue) the gitLab repo)
was adapted to run in the Gaudi software framework and to support `EDM4hep` data format for inputs and outputs.

Currently only the version of the algorithm in C++ is available, the GPU version will added soon
directly from the original standalone repository.

## Setup the environment

The following setup is considering an lxplus machine.

```bash
# source key4hep environment
source /cvmfs/sw.hsf.org/key4hep/setup.sh

# then setup this project
git clone --recurse-submodules https://github.com/key4hep/k4Clue.git
cd k4Clue
cmake -S . -B build
cmake --build build

# if installation is needed
mkdir install
cd build/ ; cmake .. -DCMAKE_INSTALL_PREFIX=../install; make install
```

## Setup CLUE

### Input parameters

CLUE needs three parameters as input:

* `dc` is the critical distance used to compute the local density;
* `rhoc` is the minimum local density for a point to be promoted as a seed;
* `outlierDeltaFactor` is  a multiplicative constant to be applied to `dc`.

(
In the original article and implementation, four parameters were needed (`dc`, `rhoc`, `deltao` and `deltac`):
* `deltao` is the maximum distance for a point to be linked to a nearest higher
point.
* `deltac` is the minimum distance for a local high density point to be promoted
as a Seed. 
)

### Detector layer layout

CLUE uses a spatial index to access and query spatial data points efficiently.
Thus, a multi-layer tessellation is created which divides the 2D space into fixed rectangular bins.
The limits and size of the tessellated space is defined by the user.

An example can be found in [LayerTilesConstants.h](include/LayerTilesConstants.h).
A step-by-step guide to introduce a new detector can be found in [another readme](include/readme.md).

## Examples of use

### CLUE as Gaudi algorithm

If the projects compiles without errors, you can go run the CLUE algorithm by
```bash
cd build/
./run gaudirun.py ../gaudi_opts/clue_gaudi_wrapper.py
```

CLUE parameters and input/output file name are contained in `gaudi_opts/clue_gaudi_wrapper.py`.
The input files are using the EDM4HEP data format and the `ECALBarrel` and `ECALEndcap` CalorimeterHit collections are required.

The output file `output.root` contains `CLUEClusters` (currently also transformed as CaloHits in `CLUEClustersAsHits`).

A simple recipe to run k4CLUE as part of the CLIC reconstruction chain can be found [here](docs/clic-recipe.md).

## Package maintainer

If you encounter any error when compiling or running this project, please contact:
* Erica Brondolin, erica.brondolin@cern.ch



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

The CLUEstering algorithm ([here](https://gitlab.cern.ch/kalos/CLUEstering) the gitLab repo)
was adapted to run in the Gaudi software framework and to support `EDM4hep` data format for inputs and outputs.

CLUEstering uses the [alpaka](https://github.com/alpaka-group/alpaka) library to run both on CPUs and GPUs, currently NVIDIA and AMD GPUs are supported. The alpaka library is fetched directly by CLUEstering and downloaded if not available. It requires Boost >1.75 to work.

CLUEstering is able to perform the clustering in different number of dimensions, here it has been specialised for 2D and 3D clustering.

## Setup the environment

The following setup is considering a machine with `cvmfs` and OS compatible with the key4hep environment.

```bash
# source key4hep environment
source /cvmfs/sw.hsf.org/key4hep/setup.sh

# then setup this project
git clone --recurse-submodules https://github.com/key4hep/k4Clue.git
cd k4Clue
cmake -S . -B build # -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.2
cmake --build build
# if installation is needed
mkdir install
cd build/ ; cmake .. -DCMAKE_INSTALL_PREFIX=../install; make install
```

## Setup CLUE

### Input parameters

CLUEstering needs three parameters as input:

* `dc` is the critical distance used to compute the local density;
* `rhoc` is the minimum local density for a point to be promoted as a seed;
* `dm` is the maximum distance considered to search for followers.

(
In the original [article](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2020.591315/full) and implementation, four parameters were needed (`dc`, `rhoc`, `deltao` and `deltac`):
* `deltao` is the maximum distance for a point to be linked to a nearest higher
point.
* `deltac` is the minimum distance for a local high density point to be promoted
as a Seed.
)

### Detector layout

CLUEstering uses a spatial index to access and query spatial data points efficiently.
Thus, a multi-layer tessellation is created which divides the space into fixed bins.
The limits of the tessellated space are defined by the algorithm based on the points in input, while the size of the bins can be chosen by the user (as a fourth input parameter to the algorithm).
This algorithm is detector agnostic, therefore it does not need any information on the detector geometry to run.

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
* Marco Rovere, marco.rovere@cern.ch
* Felice Pantaleo, felice.pantaleo@cern.ch
* Aurora Perego, aurora.perego@cern.ch

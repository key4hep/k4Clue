# Standalone CLUE Algorithm on GPU and CPU

Z.Chen[1], A. Di Pilato[2,3], F. Pantaleo[4], M. Rovere[4], C. Seez[5]

*[1] Northwestern University, [2]University of Bari, [3]INFN, [4] CERN, [5]Imperial College London*

## 1. Setup

The pre-requisite dependencies are `>=gcc7`, `>=cuda10`, `Boost`, `TBB`. Fork this repo if developers.

* **On a CERN machine with GPUs:** Source the LCG View containing the correct version of GCC and Boost:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_96/x86_64-centos7-gcc8-opt/setup.sh

# then setup this project
git clone --recurse-submodules https://gitlab.cern.ch/kalos/clue.git
cd clue
make
```

* **On an Ubuntu machine with GPUs:** Install Boost and TBB first.
```bash
sudo apt-get install libtbb-dev
sudo apt-get install libboost-all-dev

# then setup this project
git clone --recurse-submodules https://gitlab.cern.ch/kalos/clue.git
cd clue
make
```

### 2. Run CLUE
CLUE needs four parameters `dc, deltao, deltac, rhoc`

_dc_ is the critical distance used to compute the local density.
_deltao_ is the maximum distance for a point to be linked to a nearest higher
point.
_deltac_ is the minimum distance for a local high density point to be promoted
as a Seed.
_rhoc_ is the minimum local density for a point to be promoted as a Seed.

If the projects compiles without errors, you can go run the CLUE algorithm by
```bash
# ./main [fileName] [dc] [deltao] [deltac] [rhoc] [useGPU] [totalNumberOfEvent] [verbose]
./main aniso_1000 20 20 50 50 0 10 1
```

The input files are `data/input/*.csv` with columns 
* x, y, layer, weight

The output files are `data/output/*.csv` with columns
* x, y, layer, weight, rho, delta, nh, isSeed, clusterId

If you encounter any error when compiling or running this project, please
contact us.

## 3. Examples
The clustering result of a few sythetic dataset is shown below
<p align=center><img width="100%" src=https://github.com/ZihengChen/CLUEAlgo/blob/master/plots/Figure3.png/></p> 

## 4. Performance on Toy Events
We generate toy events on toy detector consist of 100 layers.
The average execution time of toy events on CPU and GPU are shown below
<p align=center><img width="80%" src=https://github.com/ZihengChen/CLUEAlgo/blob/master/plots/Figure5_1.png/></p> 


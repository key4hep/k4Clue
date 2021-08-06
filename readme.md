![Logo](plots/clue_logo.png)

# Standalone CLUE Algorithm on GPU and CPU

Z.Chen[1], A. Di Pilato[2,3], F. Pantaleo[4], M. Rovere[4], C. Seez[5]

*[1] Northwestern University, [2]University of Bari, [3]INFN, [4] CERN, [5]Imperial College London*

## 1. Setup

The pre-requisite dependencies are `>=gcc7`, `<=gcc8.3`, `>=cuda10`, `Boost`, `TBB`. Fork this repo if developers.

* **On a CERN machine:** Source the LCG View containing the correct version of GCC and Boost:
```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh

# then setup this project
git clone --recurse-submodules https://gitlab.cern.ch/kalos/clue.git
cd clue
cmake -S . -B build
cmake --build build

# if installation is needed
mkdir install
cd build/ ; cmake .. -DCMAKE_INSTALL_PREFIX=../install; make install
```
If GPU/nvcc are available in the machine, the GPU version of CLUE will also be installed.
The path to the nvcc compiler can be changed in `CmakeLists.txt` if needed.

* **On an Ubuntu machine with GPUs:** Install Boost and TBB first.
```bash
sudo apt-get install libtbb-dev
sudo apt-get install libboost-all-dev

# then setup this project
git clone --recurse-submodules https://gitlab.cern.ch/kalos/clue.git
cd clue
cmake -S . -B build
cmake --build build

# if installation is needed
mkdir install
cd build/ ; cmake .. -DCMAKE_INSTALL_PREFIX=../install; make install
```

### 2. Run CLUE
CLUE needs three parameters: `dc`, `rhoc` and `outlierDeltaFactor` (in the past four parameters were needed: `dc`, `deltao`, `deltac` and `rhoc`)

_dc_ is the critical distance used to compute the local density.
_rhoc_ is the minimum local density for a point to be promoted as a Seed.
_outlierDeltaFactor_ is  a multiplicative constant to be applied to `dc`.

( _deltao_ is the maximum distance for a point to be linked to a nearest higher
point.
 _deltac_ is the minimum distance for a local high density point to be promoted
as a Seed. )

If the projects compiles without errors, you can go run the CLUE algorithm by
```bash
cd build/src
# ./main [fileName] [dc] [rhoc] [outlierDeltaFactor] [useParallel] [verbose]
./main ../../data/input/aniso_1000.csv 20 25 2 0 1 1
```

The input files are `data/input/*.csv` with columns 
* x, y, layer, weight

The output files are `data/output/*.csv` with columns
* x, y, layer, weight, rho, delta, nh, isSeed, clusterId

If you encounter any error when compiling or running this project, please
contact us.

## 3. Examples
The clustering result of a few synthetic dataset is shown below
![Datasets](Figure3.png)

## 4. Performance on Toy Events
We generate toy events on toy detector consist of 100 layers.
The average execution time of toy events on CPU and GPU are shown below
![Execution Time](Figure5_1.png)

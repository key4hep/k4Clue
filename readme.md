[![linux](https://github.com/key4hep/k4Clue/actions/workflows/test.yml/badge.svg)](https://github.com/key4hep/k4Clue/actions/workflows/test.yml)

![Logo](plots/k4Clue_logo.png)

# k4CLUE on GPU and CPU

The CLUE algorithm ([here](https://gitlab.cern.ch/kalos/clue) the gitLab repo)
was adapted to run in the Gaudi software framework and to support EDM4hep data format for inputs and outputs.

## 1. Setup

### On a lxplus machine:

If CUDA/nvcc are found on the machine, the compilation is performed automatically also for the GPU case.
The path to the nvcc compiler will be automatically taken from the machine. In this case, `>=cuda10` and `>=gcc11` are also required.

```bash
# source key4hep environment
source /cvmfs/sw.hsf.org/key4hep/setup.sh

# get nvcc 11.4, if needed
source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.4/x86_64-centos8/setup.sh

# then setup this project
git clone --recurse-submodules https://github.com/key4hep/k4Clue.git
cd k4Clue
cmake -S . -B build
cmake --build build

# if installation is needed
mkdir install
cd build/ ; cmake .. -DCMAKE_INSTALL_PREFIX=../install; make install
```

## 2. Run CLUE standalone
CLUE needs three parameters: `dc`, `rhoc` and `outlierDeltaFactor` (in the past four parameters were needed: `dc`, `deltao`, `deltac` and `rhoc`):

* `dc` is the critical distance used to compute the local density;
* `rhoc` is the minimum local density for a point to be promoted as a seed;
* `outlierDeltaFactor` is  a multiplicative constant to be applied to `dc`.

( _deltao_ is the maximum distance for a point to be linked to a nearest higher
point.
 _deltac_ is the minimum distance for a local high density point to be promoted
as a Seed. )

### Standalone CLUE

If the projects compiles without errors, you can go run the CLUE algorithm by
```bash
# ./build/src/clue/main [fileName] [dc] [rhoc] [outlierDeltaFactor] [useParallel] [verbose] [NumTBBThreads]
./build/src/clue/main data/input/aniso_1000.csv 20 25 2 0 1 1

#in case of only CPU
#./build/src/clue_tbb_cupla/mainCuplaCPUTBB data/input/aniso_1000.csv 20 25 2 0 1 1
```

The input files are `data/input/*.csv` with columns 
* x, y, layer, weight

The output files are `data/output/*.csv` with columns
* x, y, layer, weight, rho, delta, nh, isSeed, clusterId

### CLUE as Gaudi algorithm

If the projects compiles without errors, you can go run the CLUE algorithm by
```bash
cd build/
./run gaudirun.py ../gaudi_opts/clue_gaudi_wrapper.py
```

The input files are `data/input/*.root` with data in the EDM4HEP format 
* `ECALBarrel` and `ECALEndcap` CalorimeterHit collections are required
CLUE parameters and input/output file name are contained in `clue_gaudi_wrapper.py`.

The output file `output.root` contains `CLUEClusters` (currently also transformed as CaloHits).

## 3. Run CLUE during the CLIC reconstruction

The CLIC electromagnetic calorimeter is foreseen to be a sampling calorimeter with high
granularity and is thus particularly well-suited to test the CLUE algorithm. 

A simple recipe follows to run k4CLUE as part of the CLIC reconstruction chain:
```
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
git clone --recurse-submodules https://github.com/key4hep/k4Clue.git

git clone git@github.com:key4hep/k4MarlinWrapper.git
git clone https://github.com/iLCSoft/CLICPerformance

cd CLICPerformance/clicConfig
ddsim --steeringFile clic_steer.py --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --enableGun --gun.distribution uniform --gun.particle gamma --gun.energy 10*GeV --outputFile gamma_10GeV_edm4hep.root --numberOfEvents 10

cp ../../k4MarlinWrapper/test/gaudi_opts/clicRec_e4h_input.py .
k4run clicRec_e4h_input.py --EventDataSvc.input gamma_10GeV_edm4hep.root

#You can still visualise the output in slcio with:
ced2go -d ../Visualisation/CLIC_o3_v06_CED/CLIC_o3_v06_CED.xml -s 1 Output_REC.slcio

#Run CLUE in CLIC reconstruction
cp ../../k4Clue/gaudi_opts/clicRec_e4h_input_clue.py .
k4run clicRec_e4h_input_clue.py

#Run CLUE standalone
cp ../../k4Clue/gaudi_opts/clue_gaudi_wrapper.py .
k4run clue_gaudi_wrapper.py --EventDataSvc.input my_output.root --out.filename output_clue_standalone.root
```

In case you have changed something from the original repo and you have rebuild the package, you should use `source build/clueenv.sh` to make `k4run` aware of your new changes.

## Package maintainer(s)

If you encounter any error when compiling or running this project, please contact:
* Erica Brondolin, erica.brondolin@cern.ch



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
# Run CLUE during the CLIC reconstruction

The CLIC electromagnetic calorimeter is foreseen to be a sampling calorimeter with high
granularity and is thus particularly well-suited to test the CLUE algorithm.

A simple recipe follows to run k4CLUE as part of the CLIC reconstruction chain:
```bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
git clone --recurse-submodules https://github.com/key4hep/k4Clue.git

git clone git@github.com:key4hep/k4MarlinWrapper.git
git clone https://github.com/iLCSoft/CLICPerformance

cd CLICPerformance/clicConfig
ddsim --steeringFile clic_steer.py --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --enableGun --gun.distribution uniform --gun.particle gamma --gun.energy 10*GeV --outputFile gamma_10GeV_edm4hep.root --numberOfEvents 10

cp ../../k4MarlinWrapper/test/gaudi_opts/clicRec_e4h_input.py .
k4run clicRec_e4h_input.py --EventDataSvc.input gamma_10GeV_edm4hep.root

#Run CLUE in CLIC reconstruction
cp ../../k4Clue/gaudi_opts/clicRec_e4h_input_clue.py .
k4run clicRec_e4h_input_clue.py --EventDataSvc.input gamma_10GeV_edm4hep.root

#Run CLUE standalone
cp ../../k4Clue/gaudi_opts/clue_gaudi_wrapper.py .
k4run clue_gaudi_wrapper.py --EventDataSvc.input my_output.root
```

In case you have changed something from the original repo and you have rebuild the package, you should use `source build/k4clueenv.sh` to make `k4run` aware of your new changes.

## Simulation from the detector surface

The `--enableGun` option in the `ddsim` command generates particle from the interaction vertex.

To generate particle from the suface of the CLICdet ECAL, substitute the generation command with the following
```bash
ddsim --steeringFile clic_steer.py --compactFile $LCGEO/CLIC/compact/CLIC_o3_v14/CLIC_o3_v14.xml --enableG4GPS --runType "run" --macroFile  myGPS.mac --outputFile gps_gamma_10GeV_edm4hep.root
```
where `myGPS.mac` can be found in this folder.


## Visualization

If you want to visualise the output with the CED event display:
```bash
cd ../..
glced &
k4run k4MarlinWrapper/k4MarlinWrapper/examples/event_display.py --EventDataSvc.input=CLICPerformance/clicConfig/gamma_10GeV_edm4hep.root
```

The recipe is taken from the [following webpage](https://key4hep.github.io/key4hep-doc/k4marlinwrapper/doc/starterkit/k4MarlinWrapperCLIC/CEDViaWrapper.html).


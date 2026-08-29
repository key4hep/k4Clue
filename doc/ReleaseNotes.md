# v01-01-01

* 2026-08-29 SanghyunKo ([PR#83](https://github.com/key4hep/k4Clue/pull/83))
  - Fix `calculatePosition` so that it fills physically meaningful values

* 2026-08-18 Juan Miguel Carceller ([PR#82](https://github.com/key4hep/k4Clue/pull/82))
  - Fix passing the right collection to `fillFinalClusters`. The problem is that `fillFinalClusters` uses `resolveIndex(collOffsets, index)` to map from a global index back to a collection and local index. But when processing per-collection, the indices in `clue_hit_coll_tmp` are local to that single collection, not global across all collections (for example, Collection 0 has 100 hits and then Collection 1 has 50 hits, then asking for index 30 for collection 1 will gives us index 30 of collection 0 because resolveIndex will think we are in collection 0). The fix is to create a vector containing only the current collection being processed.
  - Make `hasMaxEnergy` be always a valid index (think of the counter example when every hit has energy 0 to see why the previous code sets it to index 0, which may not be a valid index)
  - Avoid divisions and logarithms of zero
  - Initialize class members

* 2026-07-26 Juan Miguel Carceller ([PR#81](https://github.com/key4hep/k4Clue/pull/81))
  - Consolidate the CPU, CUDA, and HIP CMake target definitions in src/CMakeLists.txt.

* 2026-07-26 Juan Miguel Carceller ([PR#80](https://github.com/key4hep/k4Clue/pull/80))
  - Remove old data files
  - Remove a readme in the include folder
  - Change the location of the logo to a new logo folder
  - Move the contents of docs into doc to have a single folder for documentation

* 2026-06-17 AuroraPerego ([PR#79](https://github.com/key4hep/k4Clue/pull/79))
  - Enable the possibility to pass multiple collections to k4Clue, without the Barrel / Endcap separation
  - Add the possibility to choose how to cluster these collections: all together, one at a time, divided per detector region  
  - Implemented polar coordinates and the possibility to choose between those and Cartesian coordinates
  - Implemented 4D clustering with weighted Euclidean metric
  - refactor the `CLUECalorimeterHit` data format internally
  - Add the option to not save the `CLUEClustersAsHIts` collection

* 2026-05-13 Thomas Madlener ([PR#78](https://github.com/key4hep/k4Clue/pull/78))
  - Set the CellID encoding string in `initialize` as doing it during the event loop will no longer work (see [key4hep/k4FWCore#400](https://github.com/key4hep/k4FWCore/pull/400)
  - Switch to the newly available utilities for setting the cell id encoding (key4hep/k4FWCore#391](https://github.com/key4hep/k4FWCore/pull/391))
  - Propagate the cell id encoding of the input collection downstream instead of using a hardcoded collection name.

# v01-01-00

* 2026-01-07 AuroraPerego ([PR#74](https://github.com/key4hep/k4Clue/pull/74))
  - Remove ECAL-specific `cout` as the code can also run in HCAL
  - Make the hits collection name configurable
  - Take the clusters collection name from the input collection instead of the configuration for consistency

* 2025-12-16 Juan Miguel Carceller ([PR#75](https://github.com/key4hep/k4Clue/pull/75))
  - Exclude release notes from the license headers for pre-commit

* 2025-11-25 Juan Miguel Carceller ([PR#73](https://github.com/key4hep/k4Clue/pull/73))
  - Bump the minimum required version of CMake to 3.12

# v01-00-09

* 2025-11-05 Juan Miguel Carceller ([PR#72](https://github.com/key4hep/k4Clue/pull/72))
  - Add a file with release notes for automatic parsing that will fetch the content between

* 2025-10-31 AuroraPerego ([PR#71](https://github.com/key4hep/k4Clue/pull/71))
  - move k4Clue to Gaudi functional algorithm
  - use `IOSvc` in `clue_gaudi_wrapper.py` instead of the deprecated `PodioInput/Output`
  - added `CellIDEncoding` string to CLUE clusters and CLUE calo hits

* 2025-09-25 AuroraPerego ([PR#70](https://github.com/key4hep/k4Clue/pull/70))
  - Replace the current implementation with a new one that uses the `CLUEstering` library as an external dependency. It enables clustering in 2D and 3D and, being based on the alpaka library, allows us to run on GPU as well. Everything is configurable directly in the Python configuration file.
  - Added `cmake` files to compile for CUDA and HIP.
  - Changed the time parameters in the Python configuration file `clicRec_e4h_input_gun_clue.py` to have the local time assigned to calorimeter hits.
  - Added the associators between CLUE clusters and MC Particles.
  - Save the time information and the associators in the Ntuplizer and enable it for multi-particle events as well.
  - Add clusters position error.
  - Removed the old implementation.


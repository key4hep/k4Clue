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


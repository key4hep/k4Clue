#include "ClueGaudiAlgorithmWrapper.h"

#include "read_events.h"
#include "CLUEAlgo.h"

DECLARE_COMPONENT(ClueGaudiAlgorithmWrapper)

ClueGaudiAlgorithmWrapper::ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* pSL) :
  GaudiAlgorithm(name, pSL) {}

StatusCode ClueGaudiAlgorithmWrapper::initialize() {
  std::cout << "ClueGaudiAlgorithmWrapper::initialize()\n";
  return Algorithm::initialize();
}

void ClueGaudiAlgorithmWrapper::runAlgo( std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight,
              std::string outputFileName,
              float dc, float rhoc, float outlierDeltaFactor,
              bool verbose  ) {

  //////////////////////////////
  // run CLUE algorithm
  //////////////////////////////
  std::cout << "Using CLUEAlgo ... " << std::endl;
  CLUEAlgo clueAlgo(dc, rhoc, outlierDeltaFactor, verbose);
  clueAlgo.setPoints(x.size(), &x[0],&y[0],&layer[0],&weight[0]);
  // measure excution time of makeClusters
  auto start = std::chrono::high_resolution_clock::now();
  clueAlgo.makeClusters();
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "Elapsed time: " << elapsed.count() *1000 << " ms\n";
  // output result to outputFileName. -1 means all points.
  if(verbose)
    clueAlgo.verboseResults(outputFileName, -1);

  std::cout << "Finished running CLUE algorithm" << std::endl;
}

StatusCode ClueGaudiAlgorithmWrapper::execute() {
  std::cout << "ClueGaudiAlgorithmWrapper::execute()\n";

  //////////////////////////////
  // Read data and run algo
  //////////////////////////////
  DataHandle<edm4hep::CalorimeterHitCollection> calo_handle {  
    "EE_CaloHits_EDM4hep", Gaudi::DataHandle::Reader, this};

  const auto calo_coll = calo_handle.get();
  read_EDM4HEP_event(calo_coll, x, y, layer, weight);

  runAlgo(x, y, layer, weight, "ciao.csv", 10, 10, 2, true);

  //Cleaning
  x.clear();
  y.clear();
  layer.clear();
  weight.clear();
  
  return StatusCode::SUCCESS;
}

StatusCode ClueGaudiAlgorithmWrapper::finalize() {
  std::cout << "ClueGaudiAlgorithmWrapper::finalize()\n";
  return Algorithm::finalize();
}

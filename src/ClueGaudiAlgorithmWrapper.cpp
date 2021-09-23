#include "ClueGaudiAlgorithmWrapper.h"

#include "read_events.h"
#include "CLUEAlgo.h"

DECLARE_COMPONENT(ClueGaudiAlgorithmWrapper)

ClueGaudiAlgorithmWrapper::ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* pSL) :
  GaudiAlgorithm(name, pSL) {
  declareProperty("CriticalDistance", dc, "Used to compute the local density");
  declareProperty("MinLocalDensity", rhoc, "Minimum local density for a point to be promoted as a Seed");
  declareProperty("OutlierDeltaFactor", outlierDeltaFactor, "Multiplicative constant to be applied to CriticalDistance");
  declareProperty("OutClusters", clustersHandle, "Clusters collection (output)");
}

StatusCode ClueGaudiAlgorithmWrapper::initialize() {
  std::cout << "ClueGaudiAlgorithmWrapper::initialize()\n";
  return Algorithm::initialize();
}

std::map<int, std::vector<int> > ClueGaudiAlgorithmWrapper::runAlgo( std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight,
              std::string outputFileName,
              bool verbose  ) {

  // Run CLUE
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

  std::map<int, std::vector<int> > clueClusters = clueAlgo.getClusters();
  return clueClusters;
}

StatusCode ClueGaudiAlgorithmWrapper::execute() {
  std::cout << "ClueGaudiAlgorithmWrapper::execute()\n";

  // Read data and run algo
  DataHandle<edm4hep::CalorimeterHitCollection> EB_calo_handle {  
    "EB_CaloHits_EDM4hep", Gaudi::DataHandle::Reader, this};

  const auto EB_calo_coll = EB_calo_handle.get();
  read_EDM4HEP_event(EB_calo_coll, x, y, layer, weight);

  DataHandle<edm4hep::CalorimeterHitCollection> EE_calo_handle {  
    "EE_CaloHits_EDM4hep", Gaudi::DataHandle::Reader, this};

  const auto EE_calo_coll = EE_calo_handle.get();
  read_EDM4HEP_event(EE_calo_coll, x, y, layer, weight);
  std::cout << EB_calo_coll->size() << " caloHits in Barrel." << std::endl;
  std::cout << EE_calo_coll->size() << " caloHits in Endcap." << std::endl;

  // Put the CaloHits together
  //const auto calo_coll = EB_calo_handle.get();
  //std::copy(calo_coll->begin(), calo_coll->end(), std::back_inserter(EE_calo_coll) );
  //std::cout << calo_coll->size() << std::endl;

  std::map<int, std::vector<int> > clueClusters = runAlgo(x, y, layer, weight, "ciao.csv", true);

  // Save clusters
  edm4hep::CalorimeterHitCollection* finalClusters = clustersHandle.createAndPut();
  computeClusters(EE_calo_coll, clueClusters, finalClusters);
  std::cout << "Saved " << finalClusters->size() << " clusters" << std::endl;

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

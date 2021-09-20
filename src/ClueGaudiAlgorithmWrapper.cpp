#include "ClueGaudiAlgorithmWrapper.h"
#include "read_events.h"

DECLARE_COMPONENT(ClueGaudiAlgorithmWrapper)

ClueGaudiAlgorithmWrapper::ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* pSL) :
  GaudiAlgorithm(name, pSL) {}

StatusCode ClueGaudiAlgorithmWrapper::initialize() {
  std::cout << "ClueGaudiAlgorithmWrapper::initialize()\n";
  return Algorithm::initialize();
}

StatusCode ClueGaudiAlgorithmWrapper::execute() {
  std::cout << "ClueGaudiAlgorithmWrapper::execute()\n";

//  DataHandle<edm4hep::SimCalorimeterHitCollection> simcalo_handle {  
//    "ECalEndcapCollection", Gaudi::DataHandle::Reader, this};
//  const auto simcalo_coll = simcalo_handle.get();
//
//  std::cout << "Printing energy for ECalEndcapCollection" << std::endl;
//  for (const auto& edm_simcalo : (*simcalo_coll)) {
//    std::cout << edm_simcalo.getEnergy() << std::endl;
//  }

  //////////////////////////////
  // Read data and run algo
  //////////////////////////////
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  DataHandle<edm4hep::CalorimeterHitCollection> calo_handle {  
    "EE_CaloHits_EDM4hep", Gaudi::DataHandle::Reader, this};

  const auto calo_coll = calo_handle.get();
  read_EDM4HEP_event(calo_coll, x, y, layer, weight);

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

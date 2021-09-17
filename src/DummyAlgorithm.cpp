#include "DummyAlgorithm.h"

DECLARE_COMPONENT(DummyAlgorithm)

DummyAlgorithm::DummyAlgorithm(const std::string& name, ISvcLocator* pSL) :
  GaudiAlgorithm(name, pSL) {}

StatusCode DummyAlgorithm::initialize() {
  std::cout << "DummyAlgorithm::initialize()\n";
  return Algorithm::initialize();
}

StatusCode DummyAlgorithm::execute() {
  std::cout << "DummyAlgorithm::execute()\n";

  DataHandle<edm4hep::SimCalorimeterHitCollection> simcalo_handle {  
    "ECalEndcapCollection", Gaudi::DataHandle::Reader, this};
  const auto simcalo_coll = simcalo_handle.get();

  std::cout << "Printing energy for ECalEndcapCollection" << std::endl;
//  for (const auto& edm_simcalo : (*simcalo_coll)) {
//    std::cout << edm_simcalo.getEnergy() << std::endl;
//  }
    
  return StatusCode::SUCCESS;
}

StatusCode DummyAlgorithm::finalize() {
  std::cout << "DummyAlgorithm::finalize()\n";
  return Algorithm::finalize();
}

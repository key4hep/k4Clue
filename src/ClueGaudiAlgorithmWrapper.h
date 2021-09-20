#ifndef CLUE_GAUDI_ALGORITHM_WRAPPER_H
#define CLUE_GAUDI_ALGORITHM_WRAPPER_H

#include <GaudiAlg/GaudiAlgorithm.h>

// FWCore
#include <k4FWCore/DataHandle.h>

#include <edm4hep/SimCalorimeterHit.h>
#include <edm4hep/SimCalorimeterHitCollection.h>

class ClueGaudiAlgorithmWrapper : public GaudiAlgorithm {
public:
  explicit ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc);
  virtual ~ClueGaudiAlgorithmWrapper() = default;
  virtual StatusCode execute() override final;
  virtual StatusCode finalize() override final;
  virtual StatusCode initialize() override final;

};

#endif

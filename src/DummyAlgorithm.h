#ifndef DUMMY_ALGORITHM_H
#define DUMMY_ALGORITHM_H

#include <GaudiAlg/GaudiAlgorithm.h>

// FWCore
#include <k4FWCore/DataHandle.h>

#include <edm4hep/SimCalorimeterHit.h>
#include <edm4hep/SimCalorimeterHitCollection.h>

class DummyAlgorithm : public GaudiAlgorithm {
public:
  explicit DummyAlgorithm(const std::string& name, ISvcLocator* svcLoc);
  virtual ~DummyAlgorithm() = default;
  virtual StatusCode execute() override final;
  virtual StatusCode finalize() override final;
  virtual StatusCode initialize() override final;

};

#endif
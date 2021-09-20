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

  void runAlgo(std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight,
               std::string outputFileName,
               float dc, float rhoc, float outlierDeltaFactor,
               bool verbose  );

  private:
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;
};

#endif

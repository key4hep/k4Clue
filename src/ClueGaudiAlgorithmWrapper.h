#ifndef CLUE_GAUDI_ALGORITHM_WRAPPER_H
#define CLUE_GAUDI_ALGORITHM_WRAPPER_H

#include <GaudiAlg/GaudiAlgorithm.h>

// FWCore
#include <k4FWCore/DataHandle.h>

#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>

class ClueGaudiAlgorithmWrapper : public GaudiAlgorithm {
public:
  explicit ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc);
  virtual ~ClueGaudiAlgorithmWrapper() = default;
  virtual StatusCode execute() override final;
  virtual StatusCode finalize() override final;
  virtual StatusCode initialize() override final;

  std::map<int, std::vector<int> > runAlgo(std::vector<float>& x, std::vector<float>& y, 
                                           std::vector<int>& layer, std::vector<float>& weight);

  private:
  // Parameters in input
  std::string EBCaloCollectionName;
  std::string EECaloCollectionName;
  float dc;
  float rhoc;
  float outlierDeltaFactor;

  edm4hep::CalorimeterHitCollection calo_coll;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  // Collections in output
  DataHandle<edm4hep::CalorimeterHitCollection> fakeClustersHandle{"Output_hits", Gaudi::DataHandle::Writer, this};
  DataHandle<edm4hep::CalorimeterHitCollection> caloHitsOutliersHandle{"CLUEOutliers", Gaudi::DataHandle::Writer, this};
  DataHandle<edm4hep::ClusterCollection> clustersHandle{"CLUEClusters", Gaudi::DataHandle::Writer, this};

};

#endif

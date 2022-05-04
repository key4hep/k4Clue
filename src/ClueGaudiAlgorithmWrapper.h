#ifndef CLUE_GAUDI_ALGORITHM_WRAPPER_H
#define CLUE_GAUDI_ALGORITHM_WRAPPER_H

#include <GaudiAlg/GaudiAlgorithm.h>

// FWCore
#include <k4FWCore/DataHandle.h>

#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include "CLUECalorimeterHit.h"

class ClueGaudiAlgorithmWrapper : public GaudiAlgorithm {
public:
  explicit ClueGaudiAlgorithmWrapper(const std::string& name, ISvcLocator* svcLoc);
  virtual ~ClueGaudiAlgorithmWrapper() = default;
  virtual StatusCode execute() override final;
  virtual StatusCode finalize() override final;
  virtual StatusCode initialize() override final;

  void fillInputs(std::vector<clue::CLUECalorimeterHit>& clue_hits);
  std::map<int, std::vector<int> > runAlgo(std::vector<clue::CLUECalorimeterHit>& clue_hits, 
                                           bool isBarrel);
  void cleanCLUEinputs();
  void fillFinalClusters(const std::map<int, std::vector<int> > clusterMap, 
                         edm4hep::ClusterCollection* clusters);
  void calculatePosition(edm4hep::MutableCluster* cluster) ;
  void transformClustersInCaloHits(edm4hep::ClusterCollection* clusters,
                                 edm4hep::CalorimeterHitCollection* caloHits);

  private:
  // Parameters in input
  std::string EBCaloCollectionName;
  std::string EECaloCollectionName;
  const edm4hep::CalorimeterHitCollection* EB_calo_coll; 
  const edm4hep::CalorimeterHitCollection* EE_calo_coll;
  float dc;
  float rhoc;
  float outlierDeltaFactor;

  // CLUE inputs
  clue::CLUECalorimeterHitCollection clue_hit_coll;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  // PODIO data service
  ServiceHandle<IDataProviderSvc> m_eventDataSvc;
  PodioDataSvc* m_podioDataSvc;

  // Collections in output
  DataHandle<edm4hep::CalorimeterHitCollection> caloHitsHandle{"CLUEHits", Gaudi::DataHandle::Writer, this};
  DataHandle<edm4hep::ClusterCollection> clustersHandle{"CLUEClusters", Gaudi::DataHandle::Writer, this};

};

#endif

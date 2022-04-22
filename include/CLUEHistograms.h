#ifndef CLUE_HISTOGRAMS_H
#define CLUE_HISTOGRAMS_H

#include "k4FWCore/DataHandle.h"
#include "GaudiAlg/GaudiAlgorithm.h"
#include "GaudiKernel/ITHistSvc.h"

#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include <edm4hep/EventHeaderCollection.h>
#include "CLUECalorimeterHit.h"

#include "TH1F.h"
#include "TGraph.h"

class CLUEHistograms : public GaudiAlgorithm {

public:
  /// Constructor.
  CLUEHistograms(const std::string& name, ISvcLocator* svcLoc);
  /// Initialize.
  virtual StatusCode initialize();
  /// Execute.
  virtual StatusCode execute();
  /// Finalize.
  virtual StatusCode finalize();

private:
  const clue::CLUECalorimeterHitCollection* clue_calo_coll;
  std::string ClusterCollectionName;
  const edm4hep::ClusterCollection* cluster_coll; 

  std::uint64_t nSeeds_tot = 0;
  std::uint64_t nFollowers_tot = 0;
  std::uint64_t nOutliers_tot = 0;

  // PODIO data service
  ServiceHandle<IDataProviderSvc> m_eventDataSvc;
  PodioDataSvc* m_podioDataSvc;

  ITHistSvc* m_ths{nullptr};  ///< THistogram service
  TH1F* h_clusters{nullptr};
  TH1F* h_clSize{nullptr};
  TH1F* h_clEnergy{nullptr};
  TH1F* h_clLayer{nullptr};
  TH1F* h_clHitsLayer{nullptr};
  TH1F* h_clHitsEnergyLayer{nullptr};
  std::vector<std::string> graphClueNames{}; 
  std::vector<TGraph*> graphClue{}; 

  bool saveEachEvent{false};
  std::int32_t evNum;
  std::vector<std::string> graphPosNames{}; 
  std::map<const std::int32_t, std::vector<TGraph*>> graphPos{};
};

#endif  // CLUE_HISTOGRAMS_H

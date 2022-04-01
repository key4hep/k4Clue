#ifndef CLUE_HISTOGRAMS_H
#define CLUE_HISTOGRAMS_H

#include "k4FWCore/DataHandle.h"
#include "GaudiAlg/GaudiAlgorithm.h"
#include "GaudiKernel/ITHistSvc.h"

#include <edm4hep/CalorimeterHitCollection.h>
#include <edm4hep/ClusterCollection.h>
#include "CLUECalorimeterHit.h"

#include "TH1F.h"

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

  ITHistSvc* m_ths{nullptr};  ///< THistogram service
  TH1F* h_clusters{nullptr};
  TH1F* h_clSize{nullptr};
  TH1F* h_clEnergy{nullptr};
  TH1F* h_clLayer{nullptr};
  TH1F* h_clHitsLayer{nullptr};
  TH1F* h_clHitsEnergyLayer{nullptr};

};

#endif  // CLUE_HISTOGRAMS_H

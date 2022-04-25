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
  /// Destructor.
  ~CLUEHistograms() {
    delete m_event;
    delete m_region;
    delete m_layer;
    delete m_status;
    delete m_x;
    delete m_y;
    delete m_z;
    delete m_eta;
    delete m_phi;
    delete m_rho;
    delete m_delta;
  };
  /// Initialize.
  virtual StatusCode initialize();
  /// Initialize tree.
  void initializeTree();
  /// Clean tree.
  void cleanTree();
  /// Execute.
  virtual StatusCode execute();
  /// Finalize.
  virtual StatusCode finalize();


private:
  const clue::CLUECalorimeterHitCollection* clue_calo_coll;
  std::string ClusterCollectionName;
  const edm4hep::ClusterCollection* cluster_coll; 

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

  TTree* t_hits{nullptr};
  std::vector<int> *m_event = nullptr;
  std::vector<int> *m_region = nullptr;
  std::vector<int> *m_layer = nullptr;
  std::vector<int> *m_status = nullptr;
  std::vector<float> *m_x = nullptr;
  std::vector<float> *m_y = nullptr;
  std::vector<float> *m_z = nullptr;
  std::vector<float> *m_eta = nullptr;
  std::vector<float> *m_phi = nullptr;
  std::vector<float> *m_rho = nullptr;
  std::vector<float> *m_delta = nullptr;

  bool saveEachEvent{false};
  std::int32_t evNum;
  std::vector<std::string> graphPosNames{}; 
  std::map<const std::int32_t, std::vector<TGraph*>> graphPos{};
};

#endif  // CLUE_HISTOGRAMS_H

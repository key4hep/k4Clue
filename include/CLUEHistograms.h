#ifndef CLUE_HISTOGRAMS_H
#define CLUE_HISTOGRAMS_H

#include "k4FWCore/DataHandle.h"
#include "GaudiAlg/GaudiAlgorithm.h"
#include "GaudiKernel/ITHistSvc.h"

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
  ITHistSvc* m_ths{nullptr};  ///< THistogram service
  TH1F* h_clusters{nullptr};

};

#endif  // GENERATION_HEPMCHISTOGRAMS_H

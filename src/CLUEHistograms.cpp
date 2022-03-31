#include "CLUEHistograms.h"

#include "CLUECalorimeterHit.h"

DECLARE_COMPONENT(CLUEHistograms)

CLUEHistograms::CLUEHistograms(const std::string& name, ISvcLocator* svcLoc) : GaudiAlgorithm(name, svcLoc) {
}

StatusCode CLUEHistograms::initialize() {
  info() << "CLUEHistograms::initialize()" << endmsg;
  if (GaudiAlgorithm::initialize().isFailure()) return StatusCode::FAILURE;

  if (service("THistSvc", m_ths).isFailure()) {
    error() << "Couldn't get THistSvc" << endmsg;
    return StatusCode::FAILURE;
  }

  h_clusters = new TH1F("Num_clusters","Num_clusters",100, 0, 100);
  if (m_ths->regHist("/rec/Num_clusters", h_clusters).isFailure()) {
    error() << "Couldn't register clusters" << endmsg;
  }

  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::execute() {
  info() << "CLUEHistograms::execute()" << endmsg;

  DataObject* pStatus  = nullptr;
  StatusCode  scStatus = eventSvc()->retrieveObject("/Event/CLUECalorimeterHitCollection", pStatus);
  if (scStatus.isSuccess()) {
    clue::CLUECalorimeterHitCollection* chsingle = static_cast<clue::CLUECalorimeterHitCollection*>(pStatus);
    info() << "CH SIZE : " << chsingle->vect.size() << endmsg;
  } else {
    info() << "Status NOT success" << endmsg;
  }

  h_clusters->Fill(1);

  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::finalize() {
  if (GaudiAlgorithm::finalize().isFailure()) return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

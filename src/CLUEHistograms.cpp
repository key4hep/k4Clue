#include "CLUEHistograms.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

DECLARE_COMPONENT(CLUEHistograms)

CLUEHistograms::CLUEHistograms(const std::string& name, ISvcLocator* svcLoc) : GaudiAlgorithm(name, svcLoc) {
  declareProperty("ClusterCollection", ClusterCollectionName, "Collection of clusters in input");
}

StatusCode CLUEHistograms::initialize() {
  info() << "CLUEHistograms::initialize()" << endmsg;
  if (GaudiAlgorithm::initialize().isFailure()) return StatusCode::FAILURE;

  if (service("THistSvc", m_ths).isFailure()) {
    error() << "Couldn't get THistSvc" << endmsg;
    return StatusCode::FAILURE;
  }

  h_clusters = new TH1F("Clusters","Clusters",100, 0, 100);
  if (m_ths->regHist("/rec/Clusters", h_clusters).isFailure()) {
    error() << "Couldn't register Clusters hist" << endmsg;
  }

  h_clSize = new TH1F("ClustersSize","ClusterSize",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterSize", h_clSize).isFailure()) {
    error() << "Couldn't register ClusterSize hist" << endmsg;
  }

  h_clEnergy = new TH1F("ClusterEnergy","ClusterEnergy",100, 0, 0.100);
  if (m_ths->regHist("/rec/ClusterEnergy", h_clEnergy).isFailure()) {
    error() << "Couldn't register ClusterEnergy hist" << endmsg;
  }

  h_clLayer = new TH1F("ClusterLayer","ClusterLayer",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterLayer", h_clLayer).isFailure()) {
    error() << "Couldn't register ClusterLayer hist" << endmsg;
  }

  h_clHitsLayer = new TH1F("ClusterHitsLayer","ClusterHitsLayer",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterHitsLayer", h_clHitsLayer).isFailure()) {
    error() << "Couldn't register ClusterHitsLayer" << endmsg;
  }

  h_clHitsEnergyLayer = new TH1F("ClusterHitsEnergy_layer","ClusterHitsEnergy_layer",100, 0, 100);
  if (m_ths->regHist("/rec/ClusterHitsEnergy_layer", h_clHitsEnergyLayer).isFailure()) {
    error() << "Couldn't register ClusterHitsEnergy_layer hist" << endmsg;
  }
  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::execute() {
  info() << "CLUEHistograms::execute()" << endmsg;

  DataObject* pStatus  = nullptr;
  StatusCode  scStatus = eventSvc()->retrieveObject("/Event/CLUECalorimeterHitCollection", pStatus);
  if (scStatus.isSuccess()) {
    clue_calo_coll = static_cast<clue::CLUECalorimeterHitCollection*>(pStatus);
    info() << "CH SIZE : " << clue_calo_coll->vect.size() << endmsg;
  } else {
    info() << "Status NOT success" << endmsg;
  }

  // Read cluster collection
  DataHandle<edm4hep::ClusterCollection> cluster_handle {  
    ClusterCollectionName, Gaudi::DataHandle::Reader, this};
  cluster_coll = cluster_handle.get();

  // Get collection metadata cellID which is valid for both EB and EE
  auto collID = cluster_coll->getID();
  const auto cellIDstr = cluster_handle.getCollMetadataCellID(collID);
  const BitFieldCoder bf(cellIDstr);
  std::cout << cellIDstr << std::endl;

  std::uint64_t ch_layer = 0;

  h_clusters->Fill(cluster_coll->size());
  for (const auto& cl : *cluster_coll) {
    h_clEnergy->Fill(cl.getEnergy());
    h_clSize->Fill(cl.hits_size());
    for (const auto& hit : cl.getHits()) {
      ch_layer = bf.get( hit.getCellID(), "layer");
      h_clHitsLayer->Fill(ch_layer);
      h_clHitsEnergyLayer->Fill(ch_layer, hit.getEnergy());
    }
    h_clLayer->Fill(ch_layer);
  }

  return StatusCode::SUCCESS;
}

StatusCode CLUEHistograms::finalize() {
  if (GaudiAlgorithm::finalize().isFailure()) return StatusCode::FAILURE;

  return StatusCode::SUCCESS;
}

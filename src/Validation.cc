#include <iostream>

#include <TString.h>
#include <TH1F.h>
#include <TFile.h>

//EDM4HEP libraries
#include "edm4hep/ClusterCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"
#include "podio/ROOTReader.h"
#include "podio/EventStore.h"
#include "DDSegmentation/BitFieldCoder.h"
using namespace dd4hep ;
using namespace DDSegmentation ;

std::string bitFieldCoder = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16" ;
TH1F* h_clusters = new TH1F("Num_clusters","Num_clusters",100, 0, 100);
TH1F* h_clEnergy = new TH1F("Energy","Energy",100, 0, 0.100);
TH1F* h_clSize   = new TH1F("Size","Size",100, 0, 100);
TH1F* h_clustersLayer = new TH1F("Num_clusters_layer","Num_clusters_layer",100, 0, 100);
TH1F* h_clHitsLayer   = new TH1F("Num_clHits_layer","Num_clHits_layer",100, 0, 100);
TH1F* h_clEnergyLayer = new TH1F("Energy_layer","Energy_layer",100, 0, 100);
TH1F* h_outliers           = new TH1F("Outliers","Outliers",100, 0, 100);
TH1F* h_outlierHitsLayer   = new TH1F("Num_outliers_layer","Num_outliers_layer",100, 0, 100);
TH1F* h_outlierEnergyLayer = new TH1F("Outliers_energy_layer","Outliers_energy_layer",100, 0, 100);

void read_ClustersEDM4HEP_event(const edm4hep::ClusterCollection& cls,
                                const edm4hep::CalorimeterHitCollection& allHits){

  h_clusters->Fill(cls.size());
  std::vector<int> isOutlier (allHits.size());
  std::fill(isOutlier.begin(), isOutlier.end(), 1); 
  int iHit = 0;
  int nOutliers = 0;
  int nClusterHits = 0;
  int ch_layer = 0;
  for (const auto& cl : cls) {
    
    h_clEnergy->Fill(cl.getEnergy());
    h_clSize->Fill(cl.getHits().size());
    std::cout << cl.getHits().size() << " caloHits in this cluster" << std::endl;

    for (const auto& hit : cl.getHits()) {
      nClusterHits++;
      const BitFieldCoder bf(bitFieldCoder) ;
      ch_layer = bf.get( hit.getCellID(), "layer");
      h_clHitsLayer->Fill(ch_layer);
      h_clEnergyLayer->Fill(ch_layer, hit.getEnergy());
      for (iHit = 0; iHit < allHits.size(); iHit++) {
        if(hit.getCellID() == allHits.at(iHit).getCellID()){
          isOutlier.at(iHit) = 0;//allHits.remove(allHits.begin(), allHits.end(), allHit);
        }
      }

    }
    h_clustersLayer->Fill(ch_layer);

  }

  for (iHit = 0; iHit < allHits.size(); iHit++) {
    if(isOutlier.at(iHit)==1){
      nOutliers++;
      const BitFieldCoder bf(bitFieldCoder) ;
      ch_layer = bf.get( allHits.at(iHit).getCellID(), "layer");
      h_outlierHitsLayer->Fill(ch_layer);
      h_outlierEnergyLayer->Fill(ch_layer, allHits.at(iHit).getEnergy());
    }
  }
  h_outliers->Fill(nOutliers);
  std::cout << nClusterHits << " caloHits in the clusters" << std::endl;
  std::cout << nOutliers << " outliers" << std::endl;

}

int main(int argc, char *argv[]) {

  std::string inputFileName  = "";
  TString outputFileName = "";
  std::string clusterCollectionName = "";
  if (argc == 4) {
    inputFileName = argv[1];
    outputFileName = argv[2];
    clusterCollectionName = argv[3];
  } else {
    std::cout << "./validation [fileName] [outputFileName] [collectionName]" << std::endl;
    return 1;
  }

  // Read EDM4HEP data
  if(inputFileName.find(".root")!=std::string::npos){

    edm4hep::CalorimeterHitCollection calo_coll;
    TFile file((outputFileName+".root"),"recreate");

    std::cout<<"input edm4hep file: "<<inputFileName<<std::endl;
    podio::ROOTReader reader;
    reader.openFile(inputFileName);

    podio::EventStore store;
    store.setReader(&reader);
    unsigned nEvents = reader.getEntries();

    for(unsigned i=0; i<nEvents; ++i) {
      std::cout<<"reading event "<<i<<std::endl;

      const auto& EB_calo_coll = store.get<edm4hep::CalorimeterHitCollection>("EB_CaloHits_EDM4hep");
      if( EB_calo_coll.isValid() ) {
        for(const auto& calo_hit_EB : EB_calo_coll){
          calo_coll->push_back(calo_hit_EB.clone());
        }
      } else {
        throw std::runtime_error("Collection not found.");
      }

      std::cout << EB_calo_coll.size() << " caloHits in Barrel." << std::endl;
      const auto& EE_calo_coll = store.get<edm4hep::CalorimeterHitCollection>("EE_CaloHits_EDM4hep");
      if( EE_calo_coll.isValid() ) {
        for(const auto& calo_hit_EE : EE_calo_coll ){
          calo_coll->push_back(calo_hit_EE.clone());
        }
      } else {
        throw std::runtime_error("Collection not found.");
      }
      std::cout << EE_calo_coll.size() << " caloHits in Endcap." << std::endl;
      std::cout << calo_coll->size() << " caloHits in total. " << std::endl;

      const auto& clusters = store.get<edm4hep::ClusterCollection>(clusterCollectionName);
      if( clusters.isValid() ) {
        read_ClustersEDM4HEP_event(clusters, calo_coll);
      } else {
        throw std::runtime_error("Collection not found.");
      }
      std::cout << clusters.size() << " cluster(s) in input." << std::endl;
      calo_coll.clear();
      store.clear();
      reader.endOfEvent();
    }
    reader.closeFile();

    h_clustersLayer->Scale(1./nEvents);
    h_clHitsLayer->Scale(1./nEvents);
    h_clEnergyLayer->Scale(1./nEvents);
    h_outlierHitsLayer->Scale(1./nEvents);
    h_outlierEnergyLayer->Scale(1./nEvents);

    h_clusters->Write();
    h_clSize->Write();
    h_clEnergy->Write();
    h_clustersLayer->Write();
    h_clHitsLayer->Write();
    h_clEnergyLayer->Write();
    h_outliers->Write();
    h_outlierHitsLayer->Write();
    h_outlierEnergyLayer->Write();

    file.Close();

  }

  return 0;
}

#include <iostream>

#include <TString.h>
#include <TH1F.h>
#include <TFile.h>

//EDM4HEP libraries
#include "edm4hep/ClusterCollection.h"
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

void read_ClustersEDM4HEP_event(const edm4hep::ClusterCollection& cls){

  h_clusters->Fill(cls.size());
  for (const auto& cl : cls) {
    
    h_clEnergy->Fill(cl.getEnergy());
    h_clSize->Fill(cl.getHits().size());
    std::cout << cl.getHits().size() << " caloHits in this cluster" << std::endl;

    int ch_layer = 0;
    for (const auto& hit : cl.getHits()) {
      const BitFieldCoder bf(bitFieldCoder) ;
      ch_layer = bf.get( hit.getCellID(), "layer");
      auto ch_energy = hit.getEnergy();
      std::cout << "  " << ch_layer << std::endl;
      h_clEnergyLayer->Fill(ch_layer, hit.getEnergy());
      h_clHitsLayer->Fill(ch_layer);
    }
    h_clustersLayer->Fill(ch_layer);

  }

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

    TFile file((outputFileName+".root"),"recreate");

    std::cout<<"input edm4hep file: "<<inputFileName<<std::endl;
    podio::ROOTReader reader;
    reader.openFile(inputFileName);

    podio::EventStore store;
    store.setReader(&reader);
    unsigned nEvents = reader.getEntries();

    for(unsigned i=0; i<nEvents; ++i) {
      std::cout<<"reading event "<<i<<std::endl;

      const auto& clusters = store.get<edm4hep::ClusterCollection>(clusterCollectionName);
      if( clusters.isValid() ) {
        read_ClustersEDM4HEP_event(clusters);
      } else {
        throw std::runtime_error("Collection not found.");
      }
      std::cout << clusters.size() << " cluster(s) in input." << std::endl;

      store.clear();
      reader.endOfEvent();
    }
    reader.closeFile();

    h_clusters->Write();
    h_clSize->Write();
    h_clEnergy->Write();
    h_clustersLayer->Write();
    h_clHitsLayer->Write();
    h_clEnergyLayer->Write();
    file.Close();

  }

  return 0;
}

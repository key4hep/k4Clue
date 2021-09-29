#include <iostream>

//EDM4HEP libraries
#include "edm4hep/ClusterCollection.h"
#include "podio/ROOTReader.h"
#include "podio/EventStore.h"
#include "DDSegmentation/BitFieldCoder.h"
using namespace dd4hep ;
using namespace DDSegmentation ;

std::string bitFieldCoder = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16" ;

void read_ClustersEDM4HEP_event(const edm4hep::ClusterCollection& cl){

  for (const auto& ch : cl) {
    for (const auto& hit : ch.getHits()) {
      const BitFieldCoder bf(bitFieldCoder) ;
      auto ch_layer = bf.get( hit.getCellID(), "layer");
      auto ch_energy = hit.getEnergy();
      std::cout << "  " << ch_layer << std::endl;
    }
    std::cout << ch.getHits().size() << " caloHits in this cluster" << std::endl;
  }

}

int main(int argc, char *argv[]) {

  std::string inputFileName = "";
  std::string clusterCollectionName = "";
  if (argc == 3) {
    inputFileName = argv[1];
    clusterCollectionName = argv[2];
  } else {
    std::cout << "./validation [fileName] [collectionName]" << std::endl;
    return 1;
  }

  // Read EDM4HEP data
  if(inputFileName.find(".root")!=std::string::npos){

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

  }

  return 0;
}

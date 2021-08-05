// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <exception>
#include <cassert>

#define NLAYERS 100

// test data model
#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimTrackerHitCollection.h"
#include "edm4hep/CaloHitContributionCollection.h"
#include "edm4hep/SimCalorimeterHitCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/ClusterCollection.h"

// podio specific includes
#include "podio/EventStore.h"
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

template<typename ReaderT>
void read_events(const std::string& inputFileName, 
                 std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight,
                 unsigned int nEvents, bool isBarrel = false) {
  std::cout<<"input edm4hep file: "<<inputFileName<<std::endl;
  ReaderT reader;
  reader.openFile(inputFileName);

  podio::EventStore store;
  store.setReader(&reader);

  float x_tmp;
  float y_tmp;

  for(unsigned i=0; i<nEvents; ++i) {
    std::cout<<"reading event "<<i<<std::endl;
    std::string collectionLabel = "EE_CaloHits_EDM4hep";
    if(isBarrel)
      collectionLabel = "EB_CaloHits_EDM4hep";
    auto& chs = store.get<edm4hep::CalorimeterHitCollection>(collectionLabel);

    if( chs.isValid() ){

      for (auto ch : chs){

        const BitFieldCoder bf("system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16" ) ;
        auto ch_layer = bf.get( ch.getCellID(), "layer");
        auto ch_energy = ch.getEnergy();

	if(isBarrel){
	  //Barrel
	  x_tmp = ch.getPosition().z;
	  y_tmp = atan2(ch.getPosition().y, ch.getPosition().x);
	} else {
	  //Endcap
	  x_tmp = ch.getPosition().x;
	  y_tmp = ch.getPosition().y;
	}

        if(i==(nEvents-1)){
          x.push_back(x_tmp); 
          y.push_back(y_tmp); 
          layer.push_back(ch_layer); 
          weight.push_back(ch_energy); 
          std::cout << x_tmp << "," << y_tmp << "," << ch_layer << "," << ch_energy << std::endl;
        } else {
          std::cout<<"skip saving of event "<<i<<std::endl;
        }
      }
    } else {
      throw std::runtime_error("Collection 'EB_CaloHits_EDM4hep' should be present");
    }

    store.clear();
    reader.endOfEvent();
  }
  reader.closeFile();
}

void read_events_from_csv(const std::string& inputFileName,
                          std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight) {

  std::cout<<"input csv file: "<<inputFileName<<std::endl;
  // open csv file
  std::ifstream iFile(inputFileName);
  if( !iFile.is_open() ){
    std::cerr << "Failed to open the file" << std::endl;
    return;
  }

  // make dummy layers
  for (int l=0; l<NLAYERS; l++){
    std::string value = "";
    // Iterate through each line and split the content using delimeter
    while (getline(iFile, value, ',')) {
      x.push_back(std::stof(value)) ;
      getline(iFile, value, ','); y.push_back(std::stof(value));
      getline(iFile, value, ','); layer.push_back(std::stoi(value) + l);
      getline(iFile, value); weight.push_back(std::stof(value));
    }
  }
  std::cout << "Finished loading input points" << std::endl;

  iFile.close();
  return;
}


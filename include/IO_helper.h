// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <exception>
#include <cassert>

// test data model
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/ClusterCollection.h"

// podio specific includes
#include "DDSegmentation/BitFieldCoder.h"

using namespace dd4hep ;
using namespace DDSegmentation ;

void read_EDM4HEP_event(const edm4hep::CalorimeterHitCollection& calo_coll, std::string cellIDstr,
                        std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight) {

  float r_tmp;
  float eta_tmp;
  float phi_tmp;

  for (const auto& ch : calo_coll) {
    const BitFieldCoder bf(cellIDstr);
    auto ch_layer = bf.get( ch.getCellID(), "layer");
    auto ch_energy = ch.getEnergy();

    //eta,phi
    r_tmp = sqrt(ch.getPosition().x*ch.getPosition().x + ch.getPosition().y*ch.getPosition().y);
    eta_tmp = - 1. * log(tan(atan2(r_tmp, ch.getPosition().z)/2.));
    phi_tmp = atan2(ch.getPosition().y, ch.getPosition().x);

    x.push_back(eta_tmp); 
    y.push_back(phi_tmp); 
    layer.push_back(ch_layer); 
    weight.push_back(ch_energy); 
    //std::cout << eta_tmp << "," << phi_tmp << "," << ch_layer << "," << ch_energy << std::endl;
  }

  return;
}

void read_from_csv(const std::string& inputFileName,
                   std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight) {

  std::cout<<"input csv file: "<<inputFileName<<std::endl;
  // open csv file
  std::ifstream iFile(inputFileName);
  if( !iFile.is_open() ){
    std::cerr << "ERROR: Failed to open the file" << std::endl;
    return;
  }

  // make dummy layers
  for (int l=0; l<10; l++){
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




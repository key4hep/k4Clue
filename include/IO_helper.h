/*
 * Copyright (c) 2020-2023 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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




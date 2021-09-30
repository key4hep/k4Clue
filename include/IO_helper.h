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

std::string bitFieldCoder = "system:0:5,side:5:-2,module:7:8,stave:15:4,layer:19:9,submodule:28:4,x:32:-16,y:48:-16" ;

void read_EDM4HEP_event(const edm4hep::CalorimeterHitCollection& calo_coll,
                        std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight) {

  float x_tmp;
  float y_tmp;
  float r_tmp;

  for (const auto& ch : calo_coll) {
    const BitFieldCoder bf(bitFieldCoder) ;
    auto ch_layer = bf.get( ch.getCellID(), "layer");
    auto ch_energy = ch.getEnergy();

    //eta,phi
    r_tmp = sqrt(ch.getPosition().x*ch.getPosition().x + ch.getPosition().y*ch.getPosition().y);
    x_tmp = - 1. * log(tan(atan2(r_tmp, ch.getPosition().z)/2.));
    y_tmp = atan2(ch.getPosition().y, ch.getPosition().x);

    x.push_back(x_tmp); 
    y.push_back(y_tmp); 
    layer.push_back(ch_layer); 
    weight.push_back(ch_energy); 
//      std::cout << x_tmp << "," << y_tmp << "," << ch_layer << "," << ch_energy << std::endl;
  }

  return;
}

void read_from_csv(const std::string& inputFileName,
                          std::vector<float>& x, std::vector<float>& y, std::vector<int>& layer, std::vector<float>& weight) {

  std::cout<<"input csv file: "<<inputFileName<<std::endl;
  // open csv file
  std::ifstream iFile(inputFileName);
  if( !iFile.is_open() ){
    std::cerr << "Failed to open the file" << std::endl;
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

void computeClusters(const edm4hep::CalorimeterHitCollection& calo_coll,
                     const edm4hep::CalorimeterHitCollection* const EB_calo_coll,
                     const edm4hep::CalorimeterHitCollection* const EE_calo_coll,
                     const std::map<int, std::vector<int> > clusterMap, 
                     edm4hep::ClusterCollection* clusters,
                     edm4hep::CalorimeterHitCollection* finalOuliers){
  const BitFieldCoder bf(bitFieldCoder) ;

  for(auto cl : clusterMap){
    //std::cout << cl.first << std::endl;

    // outliers are saved in a different collection
    if(cl.first == -1){
      for(auto index : cl.second){
        auto ch_layer = bf.get( calo_coll.at(index).getCellID(), "layer");
        auto outlier = finalOuliers->create();
        if( index < EB_calo_coll->size() )
          outlier = EB_calo_coll->at(index).clone();
        else
          outlier = EB_calo_coll->at(index - EB_calo_coll->size()).clone();
      }
      continue;
    }

    std::map<int, std::vector<int> > clustersLayer;
    for(auto index : cl.second){
      auto ch_layer = bf.get( calo_coll.at(index).getCellID(), "layer");
      clustersLayer[ch_layer].push_back(index);
    }

    for(auto clLay : clustersLayer){
      float energy = 0.f;
      float energyErr = 0.f;
      auto position = edm4hep::Vector3f({0,0,0});

      auto cluster = clusters->create();
      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;
      //std::cout << "  layer = " << clLay.first << std::endl;

      for(auto index : clLay.second){
        //std::cout << "    " << index << std::endl;
        energy += calo_coll.at(index).getEnergy();
        energyErr += sqrt(calo_coll.at(index).getEnergyError()*calo_coll.at(index).getEnergyError());
        position.x += calo_coll.at(index).getPosition().x;
        position.y += calo_coll.at(index).getPosition().y;
        position.z += calo_coll.at(index).getPosition().z;
        if( index < EB_calo_coll->size() )
          cluster.addToHits(EB_calo_coll->at(index));
        else
          cluster.addToHits(EE_calo_coll->at(index - EB_calo_coll->size()));

        if (calo_coll.at(index).getEnergy() > maxEnergyValue) {
          maxEnergyValue = calo_coll.at(index).getEnergy();
          maxEnergyIndex = index;
        }
      }

      cluster.setEnergy(energy);
      cluster.setEnergyError(energyErr);
      // one could (should?) re-weight the barycentre with energy
      cluster.setPosition({position.x/clLay.second.size(), position.y/clLay.second.size(), position.z/clLay.second.size()});
      cluster.setType(calo_coll.at(maxEnergyIndex).getType());
    }
    clustersLayer.clear();

  }
  return;
}

void computeCaloHits(const edm4hep::CalorimeterHitCollection& calo_coll,
                     const std::map<int, std::vector<int> > clusterMap, 
                     edm4hep::CalorimeterHitCollection* clusters){

  const BitFieldCoder bf(bitFieldCoder) ;

  for(auto cl : clusterMap){
    //std::cout << cl.first << std::endl;

    // outliers are saved in a different collection
    if(cl.first == -1)
      continue;

    std::map<int, std::vector<int> > clustersLayer;
    for(auto index : cl.second){
      auto ch_layer = bf.get( calo_coll.at(index).getCellID(), "layer");
      clustersLayer[ch_layer].push_back(index);
    }

    for(auto clLay : clustersLayer){
      float energy = 0.f;
      float energyErr = 0.f;
      float time = 0.f;
      auto position = edm4hep::Vector3f({0,0,0});

      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;
      //std::cout << "  layer = " << clLay.first << std::endl;
      for(auto index : clLay.second){
        //std::cout << "    " << index << std::endl;
        energy += calo_coll.at(index).getEnergy();
        energyErr += sqrt(calo_coll.at(index).getEnergyError()*calo_coll.at(index).getEnergyError());
        position.x += calo_coll.at(index).getPosition().x;
        position.y += calo_coll.at(index).getPosition().y;
        position.z += calo_coll.at(index).getPosition().z;
        time += calo_coll.at(index).getTime();

        if (calo_coll.at(index).getEnergy() > maxEnergyValue) {
          maxEnergyValue = calo_coll.at(index).getEnergy();
          maxEnergyIndex = index;
        }
      }

      auto cluster = clusters->create();
      cluster.setEnergy(energy);
      cluster.setEnergyError(energyErr);
      // one could (should?) re-weight the barycentre with energy
      cluster.setPosition({position.x/clLay.second.size(), position.y/clLay.second.size(), position.z/clLay.second.size()});
      cluster.setCellID(calo_coll.at(maxEnergyIndex).getCellID());
      cluster.setType(calo_coll.at(maxEnergyIndex).getType());
      cluster.setTime(time/clLay.second.size());
    }
    clustersLayer.clear();

  }
  return;
}

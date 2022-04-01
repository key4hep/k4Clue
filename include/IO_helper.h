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
                     std::string cellIDstr,
                     const edm4hep::CalorimeterHitCollection* const EB_calo_coll,
                     const edm4hep::CalorimeterHitCollection* const EE_calo_coll,
                     const std::map<int, std::vector<int> > clusterMap, 
                     edm4hep::ClusterCollection* clusters){
  const BitFieldCoder bf(cellIDstr) ;

  for(auto cl : clusterMap){
    //std::cout << cl.first << std::endl;
    std::map<int, std::vector<int> > clustersLayer;
    for(auto index : cl.second){
      auto ch_layer = bf.get( calo_coll.at(index).getCellID(), "layer");
      clustersLayer[ch_layer].push_back(index);
    }

    for(auto clLay : clustersLayer){
      float energy = 0.f;
      float sumEnergyErrSquared = 0.f;
      auto position = edm4hep::Vector3f({0,0,0});

      auto cluster = clusters->create();
      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;
      //std::cout << "  layer = " << clLay.first << std::endl;
      for(auto index : clLay.second){
        //std::cout << "    " << index << std::endl;
        energy += calo_coll.at(index).getEnergy();
        sumEnergyErrSquared += pow(calo_coll.at(index).getEnergyError()/(1.*calo_coll.at(index).getEnergy()), 2);
        position.x += calo_coll.at(index).getPosition().x;
        position.y += calo_coll.at(index).getPosition().y;
        position.z += calo_coll.at(index).getPosition().z;
        if( EB_calo_coll->size() != 0){
          if( index < EB_calo_coll->size() ) {
            cluster.addToHits(EB_calo_coll->at(index));
            cluster.addToHitContributions(1.0);
          } else {
            cluster.addToHits(EE_calo_coll->at(index - EB_calo_coll->size()));
            cluster.addToHitContributions(1.0);
          }
        } else {
          cluster.addToHits(EE_calo_coll->at(index));
          cluster.addToHitContributions(1.0);
        }

        if (calo_coll.at(index).getEnergy() > maxEnergyValue) {
          maxEnergyValue = calo_coll.at(index).getEnergy();
          maxEnergyIndex = index;
        }
      }

      cluster.setEnergy(energy);
      cluster.setEnergyError(sqrt(sumEnergyErrSquared));
      // one could (should?) re-weight the barycentre with energy
      cluster.setPosition({position.x/clLay.second.size(), position.y/clLay.second.size(), position.z/clLay.second.size()});
      //JUST A PLACEHOLDER FOR NOW: TO BE FIXED
      cluster.setPositionError({0.00, 0.00, 0.00, 0.00, 0.00, 0.00});
      cluster.setType(calo_coll.at(maxEnergyIndex).getType());
    }
    clustersLayer.clear();

  }
  return;
}

void computeCaloHits(const edm4hep::CalorimeterHitCollection& calo_coll,
                     std::string cellIDstr,
                     const std::map<int, std::vector<int> > clusterMap, 
                     edm4hep::CalorimeterHitCollection* clusters){

  const BitFieldCoder bf(cellIDstr) ;

  for(auto cl : clusterMap){
    //std::cout << cl.first << std::endl;
    std::map<int, std::vector<int> > clustersLayer;
    for(auto index : cl.second){
      auto ch_layer = bf.get( calo_coll.at(index).getCellID(), "layer");
      clustersLayer[ch_layer].push_back(index);
    }

    for(auto clLay : clustersLayer){
      float energy = 0.f;
      float sumEnergyErrSquared = 0.f;
      float time = 0.f;
      auto position = edm4hep::Vector3f({0,0,0});

      unsigned int maxEnergyIndex = 0;
      float maxEnergyValue = 0.f;
      //std::cout << "  layer = " << clLay.first << std::endl;
      for(auto index : clLay.second){
        //std::cout << "    " << index << std::endl;
        energy += calo_coll.at(index).getEnergy();
        sumEnergyErrSquared += pow(calo_coll.at(index).getEnergyError()/(1.*calo_coll.at(index).getEnergy()), 2);
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
      cluster.setEnergyError(sqrt(sumEnergyErrSquared));

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

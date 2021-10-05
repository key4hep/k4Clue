#include <iostream>

#include <TString.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TGraph.h>
#include <TGraph2D.h>
#include <TFile.h>
#include <TDirectory.h>

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
TH1F* h_clSize   = new TH1F("Size","Size",100, 0, 100);
TH1F* h_clEnergy = new TH1F("Energy","Energy",100, 0, 0.100);
TH1F* h_clustersLayer = new TH1F("Num_clusters_layer","Num_clusters_layer",100, 0, 100);

TH1F* h_clHitsLayer   = new TH1F("Num_clHits_layer","Num_clHits_layer",100, 0, 100);
TH1F* h_clHitsEnergyLayer = new TH1F("Energy_layer","Energy_layer",100, 0, 100);

TH1F* h_outliers           = new TH1F("Outliers","Outliers",100, 0, 100);
TH1F* h_outliersLayer   = new TH1F("Num_outliers_layer","Num_outliers_layer",100, 0, 100);
TH1F* h_outliersEnergyLayer = new TH1F("Outliers_energy_layer","Outliers_energy_layer",100, 0, 100);

std::map<int, std::vector<TGraph*>> gPos;
std::map<int, std::vector<TGraph2D*>> gPos3D;
std::vector<TString> gNames = {"Pos_clusters_XZ", "Pos_clusters_YZ",
                               "Pos_clHits_XZ", "Pos_clHits_YZ",
                               "Pos_outliers_XZ", "Pos_outliers_YZ"};
std::vector<TString> gNames3D = {"Pos_clusters", "Pos_clHits", "Pos_outliers"};

void saveClustersAndCaloHits(const int nEvent, const edm4hep::ClusterCollection& cls,
                             const edm4hep::CalorimeterHitCollection& allHits
                             ){

  int nClusters = 0;
  int iHit = 0;
  int nOutliers = 0;
  int nClusterHits = 0;
  int ch_layer = 0;

  h_clusters->Fill(cls.size());
  std::vector<int> isOutlier (allHits.size());
  std::fill(isOutlier.begin(), isOutlier.end(), 1); 

  for (const auto& cl : cls) {
    
    h_clEnergy->Fill(cl.getEnergy());
    h_clSize->Fill(cl.getHits().size());
    gPos[nEvent][0]->SetPoint(nClusters, cl.getPosition().z, cl.getPosition().x);
    gPos[nEvent][1]->SetPoint(nClusters, cl.getPosition().z, cl.getPosition().y);
    gPos3D[nEvent][0]->SetPoint(nClusters, cl.getPosition().z, cl.getPosition().x, cl.getPosition().y);

    //std::cout << cl.getHits().size() << " caloHits in this cluster" << std::endl;
    for (const auto& hit : cl.getHits()) {
      gPos[nEvent][2]->SetPoint(nClusterHits, hit.getPosition().z, hit.getPosition().x);
      gPos[nEvent][3]->SetPoint(nClusterHits, hit.getPosition().z, hit.getPosition().y);
      gPos3D[nEvent][1]->SetPoint(nClusterHits, hit.getPosition().z, hit.getPosition().x, hit.getPosition().y);
      const BitFieldCoder bf(bitFieldCoder) ;
      ch_layer = bf.get( hit.getCellID(), "layer");
      h_clHitsLayer->Fill(ch_layer);
      h_clHitsEnergyLayer->Fill(ch_layer, hit.getEnergy());
      for (iHit = 0; iHit < allHits.size(); iHit++) {
        if(hit.getCellID() == allHits.at(iHit).getCellID()){
          isOutlier.at(iHit) = 0;//allHits.remove(allHits.begin(), allHits.end(), allHit);
        }
      }
      nClusterHits++;
    }
    h_clustersLayer->Fill(ch_layer);
    nClusters++;

  }

  for (iHit = 0; iHit < allHits.size(); iHit++) {
    if(isOutlier.at(iHit)==1){
      gPos[nEvent][4]->SetPoint(nOutliers, allHits.at(iHit).getPosition().z, allHits.at(iHit).getPosition().x);
      gPos[nEvent][5]->SetPoint(nOutliers, allHits.at(iHit).getPosition().z, allHits.at(iHit).getPosition().y);
      gPos3D[nEvent][2]->SetPoint(nOutliers, allHits.at(iHit).getPosition().z, allHits.at(iHit).getPosition().x, allHits.at(iHit).getPosition().y);
      const BitFieldCoder bf(bitFieldCoder) ;
      ch_layer = bf.get( allHits.at(iHit).getCellID(), "layer");
      h_outliersLayer->Fill(ch_layer);
      h_outliersEnergyLayer->Fill(ch_layer, allHits.at(iHit).getEnergy());
      nOutliers++;
    }
  }
  h_outliers->Fill(nOutliers);
  //std::cout << nClusterHits << " caloHits in the clusters" << std::endl;
  //std::cout << nOutliers << " outliers" << std::endl;

}

std::map<std::string, std::array<float, 2> > searchMinMax(const edm4hep::CalorimeterHitCollection& allHits){
  float minEnergyValue = 999999.f, min_x = 999999.f, min_y = 999999.f, min_z = 999999.f;
  float maxEnergyValue = -999999.f, max_x = -999999.f, max_y = -999999.f, max_z = -999999.f;
  for (auto iHit : allHits) {
    if (iHit.getEnergy() > maxEnergyValue) {
      maxEnergyValue = iHit.getEnergy();
    }
    if (iHit.getEnergy() < minEnergyValue) {
      minEnergyValue = iHit.getEnergy();
    }
    if (iHit.getPosition().x > max_x) {
      max_x = iHit.getPosition().x;
    }
    if (iHit.getPosition().x < min_x) {
      min_x = iHit.getPosition().x;
    }
    if (iHit.getPosition().y > max_y) {
      max_y = iHit.getPosition().y;
    }
    if (iHit.getPosition().y < min_y) {
      min_y = iHit.getPosition().y;
    }
    if (iHit.getPosition().z > max_z) {
      max_z = iHit.getPosition().z;
    }
    if (iHit.getPosition().z < min_z) {
      min_z = iHit.getPosition().z;
    }
  }

  std::map<std::string, std::array<float, 2> > ranges;
  ranges.insert( std::pair<std::string, std::array<float, 2> >("energy", {minEnergyValue, maxEnergyValue}) );
  ranges.insert( std::pair<std::string, std::array<float, 2> >("x", {min_x, max_x}) );
  ranges.insert( std::pair<std::string, std::array<float, 2> >("y", {min_y, max_y}) );
  ranges.insert( std::pair<std::string, std::array<float, 2> >("z", {min_z, max_z}) );

  std::cout << "Ranges used to plot: " << std::endl;
  for(auto range : ranges){
    std::cout << "  " << range.first << " = [" << range.second[0] << ", " << range.second[1] << "]" << std::endl;
  }

  return ranges;

}

int main(int argc, char *argv[]) {

  std::string inputFileName  = "";
  TString outputFileName = "";
  std::string clusterCollectionName = "";
  bool saveEachEvent = false;
  if (argc == 5) {
    inputFileName = argv[1];
    outputFileName = argv[2];
    clusterCollectionName = argv[3];
    saveEachEvent = (std::stoi(argv[4])==1)? true:false;
  } else {
    std::cout << "./validation [fileName] [outputFileName] [collectionName] [saveEachEvent]" << std::endl;
    return 1;
  }

  if(saveEachEvent)
    std::cout << "Careful! You will be saving more plots than you actually want!" << std::endl;

  // Read EDM4HEP data
  if(inputFileName.find(".root")!=std::string::npos){

    edm4hep::CalorimeterHitCollection calo_coll;
    TFile file((outputFileName+".root"),"recreate");
    TDirectory *dirMain;
    dirMain = file.mkdir("SummaryPlots");
    dirMain->cd();

    std::cout<<"input edm4hep file: "<<inputFileName<<std::endl;
    podio::ROOTReader reader;
    reader.openFile(inputFileName);

    podio::EventStore store;
    store.setReader(&reader);
    unsigned nEvents = reader.getEntries();

    int maxLayer = 100;
    TDirectory *dirEvents[nEvents];
    TGraph *hn[nEvents][maxLayer];

    for(unsigned i=0; i<nEvents; ++i) {
      std::cout<<"reading event "<<i<<std::endl;

      if(saveEachEvent){
        TString dirName = "event"+std::to_string(i);
        dirEvents[i] = file.mkdir(dirName);
        dirEvents[i]->cd();
        for (int l=0;l<maxLayer;l++) {
          TString gname = "h"+std::to_string(i)+"_L"+std::to_string(l);
          TString gtitle = "hit position for layer #"+std::to_string(l);
          hn[i][l] = new TGraph();
          hn[i][l]->SetName(gname);
          hn[i][l]->SetTitle(gtitle);
       }
       for (int iName=0;iName<gNames.size();iName++) {
         TGraph* g = new TGraph();
         gPos[i].push_back(g);
         gPos[i][iName]->SetName(gNames[iName]);
       }
       for (int iName=0;iName<gNames3D.size();iName++) {
         TGraph2D* g = new TGraph2D();
         gPos3D[i].push_back(g);
         gPos3D[i][iName]->SetName(gNames3D[iName]);
       }
      }
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
      std::map<std::string, std::array<float, 2> > minMaxParameters;
      if( clusters.isValid() ) {
        minMaxParameters = searchMinMax(calo_coll);
        saveClustersAndCaloHits(i, clusters, calo_coll);
      } else {
        throw std::runtime_error("Collection not found.");
      }
      std::cout << clusters.size() << " cluster(s) in input." << std::endl;
      calo_coll.clear();
      store.clear();
      reader.endOfEvent();

      if(saveEachEvent){
        for (int iName=0;iName<gNames.size();iName++) {
          if(gNames[iName].Contains("XZ")){
            gPos[i][iName]->GetXaxis()->SetLimits(minMaxParameters["z"][0], minMaxParameters["z"][1]);
            gPos[i][iName]->GetYaxis()->SetRangeUser(minMaxParameters["x"][0], minMaxParameters["x"][1]);
          }
          if(gNames[iName].Contains("YZ")){
            gPos[i][iName]->GetXaxis()->SetLimits(minMaxParameters["z"][0], minMaxParameters["z"][1]);
            gPos[i][iName]->GetYaxis()->SetRangeUser(minMaxParameters["y"][0], minMaxParameters["y"][1]);
          }
          gPos[i][iName]->Write();
        }
        for (int iName=0;iName<gNames3D.size();iName++) {
          //gPos3D[i][iName]->GetXaxis()->SetRangeUser(minMaxParameters["x"][0], minMaxParameters["x"][1]);
          //gPos3D[i][iName]->GetYaxis()->SetRangeUser(minMaxParameters["z"][0], minMaxParameters["z"][1]);
          //gPos3D[i][iName]->GetZaxis()->SetRangeUser(minMaxParameters["y"][0], minMaxParameters["y"][1]);
          //TH2F* h2 = new TH2F(gNames3D[iName], gNames3D[iName], 100, 100.0, 102.0, 100, 100.0, 102.0);
          //gPos3D[i][iName]->SetHistogram(h2);
          gPos3D[i][iName]->GetXaxis()->SetLimits(100.0, 102.0);
          gPos3D[i][iName]->GetYaxis()->SetRange(100.0, 102.0);
          gPos3D[i][iName]->GetZaxis()->SetRange(100.0, 102.0);
          gPos3D[i][iName]->Write();
          //h2->Write();
        }
//        for (int l=0;l<maxLayer;l++) {
//          hn[i][l]->Write();
//        }
      }
      std::cout<<std::endl;

    }
    reader.closeFile();

    dirMain->cd();
    h_clustersLayer->Scale(1./nEvents);
    h_clHitsLayer->Scale(1./nEvents);
    h_clHitsEnergyLayer->Scale(1./nEvents);
    h_outliersLayer->Scale(1./nEvents);
    h_outliersEnergyLayer->Scale(1./nEvents);

    h_clusters->Write();
    h_clSize->Write();
    h_clEnergy->Write();
    h_clustersLayer->Write();
    h_clHitsLayer->Write();
    h_clHitsEnergyLayer->Write();
    h_outliers->Write();
    h_outliersLayer->Write();
    h_outliersEnergyLayer->Write();

    file.Close();

  }

  return 0;
}

#include "tdrstyle.C"
#include "CMS_lumi.C"

#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>


void distribution(TString variable, TString energy, TString rhoc, TString of,
                  TString var_short, float xmin, float xmax, int bins, float ymax){

  TString settings = energy+"GeV_"+rhoc+"rhoc_"+of+"o";

  //gROOT->LoadMacro("tdrstyle.C");
  setTDRStyle();
  //gROOT->LoadMacro("CMS_lumi.C");
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);

  TString fileName = "/eos/user/e/ebrondol/SWAN_projects/CLUE/data/clicdet/tuningPars/nonGPS_endcapOnly/k4clue_clicdet_output_gamma_10GeV_endcap_500events_analysis_15dc_002rhoc_2o.root";
  //TString fileName = "/eos/user/e/ebrondol/SWAN_projects/CLUE/data/clicdet/tuningPars/GPS/k4clue_clicdet_output_gamma_"+energy+"GeV_uniform_500events_15dc_"+rhoc+"rhoc_"+of+"o.root";

  // Open input file
  auto f = TFile::Open(fileName);
  if (!f || f->IsZombie()) {
     return;
  }

  TNtuple *ntupleHits = (TNtuple*)f->Get("CLUEHits");

  TString histoNameFollowers = "h_"+var_short+"_Followers_"+settings;
  TH1F* h_var_Followers = new TH1F(histoNameFollowers, histoNameFollowers, bins, xmin, xmax);
  float ratio = (xmax-xmin)/bins;
  h_var_Followers->GetXaxis()->SetTitle(variable);
  if(variable=="totEnergyHits"){
    h_var_Followers->GetXaxis()->SetTitle(TString::Format("Total energy [GeV/%.2f]", ratio));
  } else if(variable=="totEnergyHits/MCEnergy"){
    h_var_Followers->GetXaxis()->SetTitle("E_{reco}/E_{MC}");
  } else if(variable=="layer"){
    h_var_Followers->GetXaxis()->SetTitle("Layer Number");
  } else if(variable=="energy"){
    h_var_Followers->GetXaxis()->SetTitle("Hit Energy [GeV]");
  }
  h_var_Followers->GetYaxis()->SetTitle("#Hits");
  h_var_Followers->GetYaxis()->SetTitleOffset(1.4);
  h_var_Followers->SetMaximum(ymax);
  h_var_Followers->SetLineColor(kBlue+1);
  h_var_Followers->SetLineWidth(2);
  h_var_Followers->SetLineStyle(3);

  TString histoNameOutliers = "h_"+var_short+"_Outliers_"+settings;
  TH1F* h_var_Outliers = new TH1F(histoNameOutliers, histoNameOutliers, bins, xmin, xmax);
  h_var_Outliers->SetLineColor(kRed+1);
  h_var_Outliers->SetLineWidth(2);
  h_var_Outliers->SetLineStyle(2);

  TString histoNameSeeds = "h_"+var_short+"_Seeds_"+settings;
  TH1F* h_var_Seeds = new TH1F(histoNameSeeds, histoNameSeeds, bins, xmin, xmax);
  h_var_Seeds->SetLineColor(kGreen+2);
  h_var_Seeds->SetLineWidth(2);
  h_var_Seeds->SetLineStyle(1);

  TCanvas *cTrash = new TCanvas();//,200,10,700,780);
  ntupleHits->Draw(variable + " >> "+histoNameFollowers, "status==1 && region ==1");
  ntupleHits->Draw(variable + " >> "+histoNameOutliers, "status==0  && region ==1");
  ntupleHits->Draw(variable + " >> "+histoNameSeeds, "status==2     && region ==1");

  //TLegend* t1 = new TLegend(0.60,0.23,0.90,0.38);
  TLegend* t1 = new TLegend(0.60,0.73,0.90,0.88);
  TString header = "endcap, o_{f} = "+of+", #rho = "+rhoc[0]+"."+rhoc[1]+rhoc[2];
  t1->SetHeader(header);
  t1->AddEntry(h_var_Seeds, "Seeds", "l");
  t1->AddEntry(h_var_Followers, "Followers", "l");
  t1->AddEntry(h_var_Outliers, "Outliers", "l");

  TLine *t500 = new TLine(xmin, 500, xmax, 500);
  t500->SetLineColor(kOrange+2);

  TString canvasName = "c_"+var_short+"_"+settings;
  TCanvas *c_var = new TCanvas(canvasName,canvasName, 50, 50, 600, 600);//,200,10,700,780);
  gStyle->SetOptStat(0);
  if(variable=="energy"){
    c_var->SetLogy();
  }
  c_var->SetRightMargin(0.06);
  c_var->cd();
  h_var_Followers->Draw();
  h_var_Outliers->Draw("same");
  h_var_Seeds->Draw("same");
  t1->Draw("same");
//  t500->Draw("same");
  c_var->Draw();
  c_var->SaveAs("../plots/"+variable+"Distr_"+settings+"_log.png", "png");

  return;

}

void hitsPlots(){

  distribution("layer", "10", "002", "2", "lay", 0, 80, 80, 1200);
//  distribution("layer", "10", "002", "1", "lay", 0, 40, 40, 5500);
//  distribution("layer", "10", "002", "3", "lay", 0, 40, 40, 5500);
//  distribution("layer", "10", "001", "2", "lay", 0, 40, 40, 5500);

//  distribution("energy", "10", "002", "2", "en", 0, 1, 100, 100000);
  return;
}

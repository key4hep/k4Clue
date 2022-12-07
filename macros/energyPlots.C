#include "tdrstyle.C"

#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTree.h>


void fitVariable(TString variable, TString energy, TString var_short, float xmin, float xmax, int bins){
  setTDRStyle();
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);

//  TString fileName = "../build/output_k4clue_analysis.root";
  TString fileName = "/eos/user/e/ebrondol/SWAN_projects/CLUE/data/clicdet/gpsProduction/k4clue_clicdet_output_gamma_"+energy+"GeV_uniform_500events_15dc_001rhoc_2o.root";

  // Open input file
  auto f = TFile::Open(fileName);
  if (!f || f->IsZombie()) {
     return;
  }

  TNtuple *ntupleCLUE = (TNtuple*)f->Get("CLUEClusters");
  TNtuple *ntuplePandora = (TNtuple*)f->Get("PandoraClusters");

  TString histoNamePandora = "h_"+var_short+"_Pandora_"+energy+"GeV";
  TH1F* h_var_Pandora = new TH1F(histoNamePandora, histoNamePandora, bins, xmin, xmax);
  float ratio = (xmax-xmin)/bins;
  h_var_Pandora->GetXaxis()->SetTitle(variable);
  if(variable=="totEnergyHits"){
    h_var_Pandora->GetXaxis()->SetTitle(TString::Format("Total energy [GeV/%.2f]", ratio));
  } else if(variable=="totEnergyHits/MCEnergy"){
    h_var_Pandora->GetXaxis()->SetTitle("E_{reco}/E_{MC}");
  }
  h_var_Pandora->GetYaxis()->SetTitle("a.u.");
  h_var_Pandora->GetYaxis()->SetTitleOffset(1.4);
  // h_var_Pandora->SetMaximum(120);
  h_var_Pandora->SetLineColor(kCyan+1);
  h_var_Pandora->SetLineWidth(2);
  h_var_Pandora->SetLineStyle(1);

  TString histoNameCLUE = "h_"+var_short+"_CLUE_"+energy+"GeV";
  TH1F* h_var_CLUE = new TH1F(histoNameCLUE, histoNameCLUE, bins, xmin, xmax);
  h_var_CLUE->SetLineColor(kRed+1);
  h_var_CLUE->SetLineWidth(1);
  h_var_CLUE->SetLineStyle(1);

  TCanvas *cTrash = new TCanvas();//,200,10,700,780);
  ntuplePandora->Draw(variable + " >> "+histoNamePandora);
  ntupleCLUE->Draw(variable + " >> "+histoNameCLUE);

  h_var_Pandora->Fit("gaus", "", "", xmin, xmax);
  TF1 *fit_var_Pandora = h_var_Pandora->GetFunction("gaus");
  fit_var_Pandora->SetLineColor(kCyan); 
  fit_var_Pandora->SetLineStyle(2);
  fit_var_Pandora->Draw("same");

  h_var_CLUE->Fit("gaus", "", "", xmin, xmax);
  TF1 *fit_var_CLUE = h_var_CLUE->GetFunction("gaus");
  fit_var_CLUE->SetLineColor(kRed+2); 
  fit_var_CLUE->SetLineStyle(2);
  fit_var_CLUE->Draw("same");

  TLegend* t1 = new TLegend(0.19,0.73,0.47,0.88);
  t1->AddEntry(h_var_Pandora, "PandoraClusters", "l");
  t1->AddEntry((TObject*)0, TString::Format("#mu = %.3f #pm %.3f", fit_var_Pandora->GetParameter(1), fit_var_Pandora->GetParError(1)), "");
  t1->AddEntry((TObject*)0, TString::Format("#sigma = %.3f #pm %.3f", fit_var_Pandora->GetParameter(2), fit_var_Pandora->GetParError(2)), "");
  t1->AddEntry(h_var_CLUE, "CLUEClusters", "l");
  t1->AddEntry((TObject*)0, TString::Format("#mu = %.3f #pm %.3f", fit_var_CLUE->GetParameter(1), fit_var_CLUE->GetParError(1)), "");
  t1->AddEntry((TObject*)0, TString::Format("#sigma = %.3f #pm %.3f", fit_var_CLUE->GetParameter(2), fit_var_CLUE->GetParError(2)), "");

  TString canvasName = "c_"+var_short+"_"+energy+"GeV";
  TCanvas *c_var = new TCanvas(canvasName,canvasName, 50, 50, 600, 600);//,200,10,700,780);
  gStyle->SetOptStat(0);
  c_var->SetRightMargin(0.06);
  c_var->cd();
  h_var_Pandora->Draw();
  h_var_CLUE->Draw("same");
  t1->Draw("same");
  c_var->Draw();
  c_var->SaveAs("../plots/"+var_short+"_"+energy+"GeV.png", "png");

  return;

}

void energyPlots(){

  fitVariable("totEnergyHits", "10", "ene", 7.5, 12.5, 25);
//  fitVariable("totEnergyHits", "20", "ene", 15.5, 25.5, 20);
//  fitVariable("totEnergyHits", "50", "ene", 40.5, 60.5, 50);
//  fitVariable("totEnergyHits", "100", "ene", 80.5, 120.5, 50);
//  fitVariable("totEnergyHits", "200", "ene", 180.5, 220.5, 50);

  fitVariable("totEnergyHits/MCEnergy", "10", "rat_ene", 0.8, 1.2, 25);
  //fitVariable("totEnergyHits/MCEnergy", "20", "rat_ene", 0.8, 1.2, 25);
  //fitVariable("totEnergyHits/MCEnergy", "50", "rat_ene", 0.8, 1.2, 25);
  //fitVariable("totEnergyHits/MCEnergy", "100", "rat_ene", 0.9, 1.1, 25);
  //fitVariable("totEnergyHits/MCEnergy", "200", "rat_ene", 0.9, 1.1, 25);

  return;
}

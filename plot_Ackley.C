#include "TF2.h"
#include "TF3.h"
#include "TError.h"
#include "TRandom3.h"



double* fcn1(double* x, double* par){
	return -20*exp(-0.2*TMath::Sqrt(0.5*(x[0]**2+x[1]**2)))-exp(0.5*(cos(2*TMath::Pi()*x[0])+cos(2*TMath::Pi()*x[1])))+20+TMath::Exp(1);
}



void plot_Ackley(){

	gStyle->SetOptStat(0);
	gStyle->SetDateY(-1);
	gStyle->SetPadGridX(0);
	gStyle->SetPadGridY(0);

	gStyle->SetTitleBorderSize(0);


//	TFile* file = new TFile("Ackley_file.root.20180609","read");
	TFile* file = new TFile("Ackley_file.root","read");

	TF2* conts[10];
	TH3F* hist[10];
	
	TF2* ref = new TF2("ref",fcn1,-3,3,-3,3,0);

	TCanvas* c = new TCanvas("c","c",600,600);
	c->cd();
	
	TLegend* leg = new TLegend(0.12,0.6,0.6,0.88);
	leg->SetBorderSize(0);
	leg->SetTextFont(42);
	leg->SetFillStyle(0);

	double level[10];
	for(int i=0;i<6;i++){
		hist[i] = (TH3F*)file->Get(Form("roothist_%d",i));
		hist[i]->SetTitle("Ackley function;x;y");
		level[i] = hist[i]->GetMean(3);
		conts[i] = new TF2(Form("f3_%d",i),fcn1,-3,3,-3,3,0); 

		if(i==0) {
			
//			ref->Draw("cont4");
			
			TH2F* tmp = (TH2F*)(hist[i]->Project3D("YX"));
			tmp->SetTitle("Ackley function;x;y");	
			tmp->GetXaxis()->SetRangeUser(-3,3);
			tmp->GetYaxis()->SetRangeUser(-3,3);
			tmp->Draw("");

	
		}
		else hist[i]->Project3D("YX")->Draw("same");
		
		conts[i]->SetLineColor(hist[i]->GetMarkerColor());
		conts[i]->SetLineStyle(8);
		conts[i]->SetContour(1,&level[i]);
		conts[i]->Draw("cont2 same");

//		if(i==0) leg->AddEntry(conts[i],Form("%d iteration roots and contour",i+1),"l");	
		leg->AddEntry(conts[i],Form("%d iteration",i+1),"l");
		leg->Draw("same");
	
	}

	c->SaveAs("~/WWW/Banach/plot_Ackley.pdf");

}

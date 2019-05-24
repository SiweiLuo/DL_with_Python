#include "TF2.h"
#include "TF3.h"
#include "TError.h"
#include "TRandom3.h"

double* fcn1(double* x, double* par){
//	return x[0]**2+x[1]**2+x[2]**2;
	double result =  sin(x[0]+x[1])+(x[0]-x[1])*(x[0]-x[1])-1.5*x[0]+2.5*x[1]+1;
	return result;
	//	return x[0]*x[0]+x[1]*x[1];
}



void plot_McCormick(){

	gStyle->SetOptStat(0);
	gStyle->SetDateY(-1);
	gStyle->SetPadGridX(0);
	gStyle->SetPadGridY(0);

	gStyle->SetTitleBorderSize(0);
	


	TFile* file = new TFile("McCormick_file.root","read");

	TF2* conts[10];
	TH3F* hist[10];
	
	TF2* ref = new TF2("ref",fcn1,-3,3,-3,3,0);

	TCanvas* c = new TCanvas("c","c",600,600);
	c->cd();
	
	TLegend* leg = new TLegend(0.12,0.6,0.6,0.88);
	leg->SetBorderSize(0);
	leg->SetTextFont(42);

	double level[10];
	for(int i=0;i<4;i++){
		hist[i] = (TH3F*)file->Get(Form("roothist_%d",i));
		hist[i]->SetTitle("McCormick function;x;y");
		level[i] = hist[i]->GetMean(3);
		conts[i] = new TF2(Form("f3_%d",i),fcn1,-3,3,-3,3,0); 

		if(i==0) {
			
//			ref->Draw("cont4");
			
			TH2F* tmp = (TH2F*)hist[i]->Project3D("YX");
			tmp->SetTitle("McCormick function;x;y");	
			tmp->Draw("");

	
		}
		else hist[i]->Project3D("YX")->Draw("same");
		
		conts[i]->SetLineColor(hist[i]->GetMarkerColor());
		conts[i]->SetLineStyle(8);
		conts[i]->SetContour(1,&level[i]);
		conts[i]->Draw("cont2 same");

		leg->AddEntry(conts[i],Form("%d iteration roots and contour",i+1),"l");	
		leg->Draw("same");
	
	}

	c->SaveAs("~/WWW/Banach/plot_McCormick.pdf");

}

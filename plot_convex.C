void plot_convex(){
	gStyle->SetTitleStyle(0);
	gStyle->SetTitleBorderSize(0);

	gStyle->SetOptStat("em");

	gStyle->SetStatY(0.8);
	gStyle->SetStatX(0.98);
	gStyle->SetStatW(0.3);
	gStyle->SetStatH(0.1);

	gStyle->SetStatBorderSize(0);
	
	gStyle->SetStatFontSize(0.1);

	gStyle->SetStatFormat("3.2g");

	gStyle->SetFillColor(-1);

	gStyle->SetTitleY(0.9);
	gStyle->SetTitleX(0.1);


	TFile* file = new TFile("optimize_20roots.root","read");

	TCanvas* c = new TCanvas("c","c",1200,300);
	c->Divide(4,1);

	TH3F* hist[4];
	TF3* fun3[4];

	for(int i=0;i<4;i++){
		c->cd(i+1);
		fun3[i] = (TF3*)file->Get(Form("f3_%d",i));
		hist[i] = (TH3F*)file->Get(Form("roothist_%d",i));
	
//		if(i==0) fun3[i]->Draw("FBBB");
//		else fun3[i]->Draw("FBBB same");
		hist[i]->Draw("FBBB");

		fun3[i]->Draw("FBBB same");
		hist[i]->Draw("FBBB same");

	}


	c->SaveAs("~/WWW/Banach/plot_convex.pdf");


}

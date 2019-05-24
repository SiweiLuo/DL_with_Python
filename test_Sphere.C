#include "RConfigure.h"

#ifdef R__HAS_MATHMORE

#include "Math/MultiRootFinder.h"
#endif

#include "Math/WrappedMultiTF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TError.h"
#include "TRandom3.h"

using namespace ROOT::Math;

double* fcn1(double* x, double* par){
	return x[0]**2+x[1]**2+x[2]**2;
}

double* fcn2(double* x, double* par){
	return x[0]**2+x[1]**2+x[2]**2-par[0];
}

TF3 * q3 = new TF3("g3",fcn1,-3,3,-3,3,-3,3,0);

TRandom3* mRan = new TRandom3();

std::vector<double> flevel; 

	double x_i[3] = {1,1,1};

	TH3F* roothist[10];
void test_Sphere(const char * algo = 0, int printlevel = 1) {
#ifndef R__HAS_MATHMORE

	Error("exampleMultiRoot","libMathMore is not available - cannot run this tutorial");

#else

	TF3 * f3 = new TF3("f3",fcn2,-3,3,-3,3,-3,3,1);
	f3->SetTitle("");

	TFile* file = new TFile("Sphere.root","recreate");
	file->cd();

	double h_i = q3->Eval(x_i[0],x_i[1],x_i[2]);
gStyle->SetTitleBorderSize(0);
gStyle->SetTitleStyle(0);

	f3->SetParameter(0,h_i);

	int i=0;
	while(1){
		flevel.push_back(iterate(f3,i));
		cout<<"flevel[flevel.size()-1] = "<<flevel[flevel.size()-1]<<"         and     "<<flevel[flevel.size()-2]<<endl;
		f3->SetParameter(0,flevel[flevel.size()-1]);
		cout<<" flevel.size()-1 "<<flevel.size()-1<<" flevel[flevel.size()-1] = "<<flevel[flevel.size()-2]-flevel[flevel.size()-1]<<endl;
		if(flevel.size()>=2 && flevel[flevel.size()-2]-flevel[flevel.size()-1]<1e-6) break; 
		++i;
	}

#endif
}

Double_t iterate(TF3* f3, Int_t it){
	ROOT::Math::MultiRootFinder r("hybrid");

	ROOT::Math::WrappedMultiTF1 g1(*f3,3);
	ROOT::Math::WrappedMultiTF1 g2(*f3,3);
	ROOT::Math::WrappedMultiTF1 g3(*f3,3);

	r.AddFunction(g1);
	r.AddFunction(g2);
	r.AddFunction(g3);

	roothist[it] = new TH3F(Form("roothist_%d",it),"roothist;x;y;z",100,-3,3,100,-3,3,100,-3,3);
	roothist[it]->SetName(Form("roothist_%d",it));
	roothist[it]->SetMarkerSize(0.5-0.05*it);

	double tolerance = 1;


	for(int i=0;i<10;){
		double x2[3]={mRan->Uniform(-3,3),mRan->Uniform(-3,3),mRan->Uniform(-3,3)};
		r.Solve(x2);
		if(fabs(f3->Eval(r.X()[0],r.X()[1],r.X()[2]))>1e-6) continue;
		roothist[it]->Fill(r.X()[0],r.X()[1],r.X()[2]);
		i++;
	}

	Double_t height = q3->Eval(r.X()[0],r.X()[1],r.X()[2]);

	Double_t level = q3->Eval(roothist[it]->GetMean(1),roothist[it]->GetMean(2),roothist[it]->GetMean(3));
	cout<<" minimum point ===> "<<roothist[it]->GetMean(1)<<"   "<<roothist[it]->GetMean(2)<<"  "<<roothist[it]->GetMean(3)<<endl;

	if(it==0) roothist[it]->SetTitle(Form("f(%.2f,%.2f,%.2f)=%.2f",x_i[0],x_i[1],x_i[2],height));
	else roothist[it]->SetTitle(Form("f(%.2f,%.2f,%.2f)=%.2f",roothist[it-1]->GetMean(1),roothist[it-1]->GetMean(2),roothist[it-1]->GetMean(3),height));



	roothist[it]->SetMarkerStyle(20);
	roothist[it]->SetMarkerColor(kRed);
	roothist[it]->Write();
	f3->SetName(Form("f3_%d",it));
	f3->SetTitle(Form("f(x) = f(x_{%d}) = %.10f",it,height));
	f3->Write();


	return level; 
}





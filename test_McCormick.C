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

//TF3 * q3 = new TF3("g3","x*x+y*y+z*z",-3,3,-3,3,-3,3);
double* fcn1(double* x, double* par){
//	return x[0]**2+x[1]**2+x[2]**2;
	double result =  sin(x[0]+x[1])+(x[0]-x[1])*(x[0]-x[1])-1.5*x[0]+2.5*x[1]+1;
	return result;
	//	return x[0]*x[0]+x[1]*x[1];
}

double* fcn2(double* x, double* par){
	return x[0]**2+x[1]**2+x[2]**2-par[0];
}


double* McCormick(double* x, double* par){

	double result = sin(x[0]+x[1])+(x[0]-x[1])*(x[0]-x[1])-1.5*x[0]+2.5*x[1]+1;
	result = result - par[0];
	return result;
	//	return x[0]*x[0]+x[1]*x[1]-par[0];
}

//TF3 * q3 = new TF3("g3",fcn1,-3,3,-3,3,-3,3,0);
TF2 * q3 = new TF2("g3",fcn1,-3,3,-3,3,0);

TRandom3* mRan = new TRandom3();

std::vector<double> flevel; 

void test_McCormick(const char * algo = 0, int printlevel = 1) {
#ifndef R__HAS_MATHMORE

	Error("exampleMultiRoot","libMathMore is not available - cannot run this tutorial");

#else

//	TF3 * f3 = new TF3("f3","x*x+y*y+z*z-[0]",-3,3,-3,3,-3,3);
//	TF3 * f3 = new TF3("f3",fcn2,-3,3,-3,3,-3,3,1);
	TF2 * f3 = new TF2("f3",McCormick,-3,3,-3,3,1);
	f3->SetTitle("");



	TFile* file = new TFile("McCormick_file.root","recreate");
	file->cd();
	q3->Write();
	
	double x_i[2] = {2,2};
	double h_i = q3->Eval(x_i[0],x_i[1]);
	//	f1->SetParameters(1,0);
	//	f2->SetParameter(0,10);


	cout<<" h_i ======== "<<h_i<<endl;	
	f3->SetParameter(0,h_i);

	int i=0;
	while(1){
		flevel.push_back(iterate(f3,i));
		cout<<"flevel[flevel.size()-1] = "<<flevel[flevel.size()-1]<<"         and     "<<flevel[flevel.size()-2]<<endl;
		f3->SetParameter(0,flevel[flevel.size()-1]);
		cout<<" flevel.size()-1 "<<flevel.size()-1<<" flevel[flevel.size()-1] = "<<flevel[flevel.size()-2]-flevel[flevel.size()-1]<<endl;
		cout<<" flevel[flevel.size()-2] ===="<<flevel[flevel.size()-2]<<endl;
		cout<<" flevel[flevel.size()-1] ===="<<flevel[flevel.size()-1]<<endl;	
		if(flevel.size()>=2 && (flevel[flevel.size()-2]-flevel[flevel.size()-1])<1e-6 && flevel[flevel.size()-1]<flevel[flevel.size()-2]) break; 
		++i;
	}

#endif
}

//Double_t iterate(TF3* f3, Int_t it){
Double_t iterate(TF2* f3, Int_t it){
	ROOT::Math::MultiRootFinder r("hybrid");
/*
	ROOT::Math::WrappedMultiTF1 g1(*f3,3);
	ROOT::Math::WrappedMultiTF1 g2(*f3,3);
	ROOT::Math::WrappedMultiTF1 g3(*f3,3);
	r.AddFunction(g1);
	r.AddFunction(g2);
	r.AddFunction(g3);


*/
	ROOT::Math::WrappedMultiTF1 g1(*f3,2);
	ROOT::Math::WrappedMultiTF1 g2(*f3,2);
	r.AddFunction(g1);
	r.AddFunction(g2);

	
	//	ROOT::Math::WrappedMultiTF1 g3(*f3,2);

	//	r.SetPrintLevel(-1);

	TH3F* roothist = new TH3F(Form("roothist_%d",it),"roothist",100,-3,3,100,-3,3,100,-4,10);
	roothist->SetName(Form("roothist_%d",it));
	roothist->SetMarkerSize(0.5-0.05*it);

	double tolerance = 1;

		cout<<" f3->GetParameter(0) == "<<f3->GetParameter(0)<<endl;

	double height;
//	for(int i=0;i<5;){
	mRan->SetSeed();

	for(int i=0;i<8;){
		double x2[2]={mRan->Uniform(-3,3),mRan->Uniform(-3,3)};
		r.Solve(x2);
		cout<<" roots ====> "<<r.X()[0]<<"   "<<r.X()[1]<<endl;
		cout<<" f3 = "<<f3->Eval(r.X()[0],r.X()[1])<<endl;
		if(fabs(f3->Eval(r.X()[0],r.X()[1]))>1e-8 || fabs(r.X()[0])>3.0 || fabs(r.X()[1])>3.0 ) continue;
//		cout<<" fabs(f3->Eval(r.X()[0],r.X()[1])) = "<<fabs(f3->Eval(r.X()[0],r.X()[1]))<<endl;
		//		roothist->Fill(r.X()[0],r.X()[1],r.X()[2]);
		
		height = q3->Eval(r.X()[0],r.X()[1]);
		cout<<" height ===== "<<height<<endl;
				roothist->Fill(r.X()[0],r.X()[1],height);
		i++;
	}


	Double_t level = q3->Eval(roothist->GetMean(1),roothist->GetMean(2));
	cout<<" minimum point ===> "<<roothist->GetMean(1)<<"   "<<roothist->GetMean(2)<<endl;
//	cout<<" height = "<<height<<" flevel = "<<flevel[flevel.size()-1]<<endl;

	roothist->SetTitle(Form("f(x) = f(x_{%d}) = %.2f, min at (%.4f,%.4f)",it,height,roothist->GetMean(1),roothist->GetMean(2)));
	roothist->SetMarkerStyle(20);
	roothist->SetMarkerColor(it+1);
	roothist->Write();
	cout<<" iteration ===== "<<it<<endl;
	cout<<" average point === "<<roothist->GetMean(1)<<"   "<<roothist->GetMean(2)<<endl;
	f3->SetName(Form("f3_%d",it));
	f3->SetTitle(Form("f(x) = f(x_{%d}) = %.10f, min at (%.4f,%.4f)",it,height,roothist->GetMean(1),roothist->GetMean(2)));
	f3->SetContour(1,&height);
	f3->SetLineColor(it+1);
	f3->Write();

	return level; 
}





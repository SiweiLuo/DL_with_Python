#include "RConfigure.h"

#ifdef R__HAS_MATHMORE

#include "Math/MultiRootFinder.h"
#endif

#include "Math/WrappedMultiTF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TError.h"
#include "TRandom3.h"
#include "roots.h"
#include "roots.cxx"

using namespace ROOT::Math;

double* fcn1(double* x, double* par){
	return -20*exp(-0.2*TMath::Sqrt(0.5*(x[0]**2+x[1]**2)))-exp(0.5*(cos(2*TMath::Pi()*x[0])+cos(2*TMath::Pi()*x[1])))+20+TMath::Exp(1);
}

double* fcn2(double* x, double* par){
	return x[0]**2+x[1]**2+x[2]**2-par[0];
}


double* Ackley(double* x, double* par){
	return -20*exp(-0.2*TMath::Sqrt(0.5*(x[0]**2+x[1]**2)))-exp(0.5*(cos(2*TMath::Pi()*x[0])+cos(2*TMath::Pi()*x[1])))+20-par[0]+TMath::Exp(1);
}

TF2 * q3 = new TF2("g3",fcn1,-3,3,-3,3,0);

TRandom3* mRan = new TRandom3();
std::vector<double> flevel; 

std::vector<double> root_average_x,root_average_y;

void test_Ackley(const char * algo = 0, int printlevel = 1) {
#ifndef R__HAS_MATHMORE

	Error("exampleMultiRoot","libMathMore is not available - cannot run this tutorial");

#else

	TF2 * f3 = new TF2("f3",Ackley,-3,3,-3,3,1);
	f3->SetTitle("");



	TFile* file = new TFile("Ackley_file.root","recreate");
	file->cd();

	cout<<" q3->Eval(0,0) ==== "<<q3->Eval(0,0)<<endl;

	q3->Write();

	double x_i[2] = {2,2};
	double h_i = q3->Eval(x_i[0],x_i[1]);

	cout<<" h_i ======== "<<h_i<<endl;	
	f3->SetParameter(0,h_i);
	
	root_average_x.push_back(x_i[0]);
	root_average_y.push_back(x_i[1]);
	flevel.push_back(h_i);

	int i=0;
	while(1){
L1:	
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

Double_t iterate(TF2* f3, Int_t it){

	mRan->SetSeed();

	ROOT::Math::MultiRootFinder r("hybrid");

	ROOT::Math::WrappedMultiTF1 g1(*f3,2);
	ROOT::Math::WrappedMultiTF1 g2(*f3,2);
	r.AddFunction(g1);
	r.AddFunction(g2);

	TH3F* roothist = new TH3F(Form("roothist_%d",it),"roothist",100,-3,3,100,-3,3,100,-10,100);
	roothist->SetName(Form("roothist_%d",it));
	roothist->SetMarkerSize(0.5-0.05*it);


	double tolerance = 1;

	cout<<" f3->GetParameter(0) == "<<f3->GetParameter(0)<<endl;

	double height;

	roots root[5];

L1:

	for(int i=0;i<5;){
		double x2[2]={mRan->Uniform(-3,3),mRan->Uniform(-3,3)};
		r.Solve(x2);
		if(fabs(q3->Eval(r.X()[0],r.X()[1])-f3->GetParameter(0))>1e-5 || fabs(r.X()[0])>3. || fabs(r.X()[1])>3.) continue;
		cout<<" f3 = "<<f3->Eval(r.X()[0],r.X()[1])<<endl;
		cout<<" q3->Eval(r.X()[0],r.X()[1]) = "<<q3->Eval(r.X()[0],r.X()[1])<<endl;
		cout<<" f3->GetParameter(0) ="<<f3->GetParameter(0)<<endl;
		cout<<" roots ====> "<<r.X()[0]<<"   "<<r.X()[1]<<endl;

		height = q3->Eval(r.X()[0],r.X()[1]);
		root[i].set_values(r.X()[0],r.X()[1]);
		cout<<" height ===== "<<height<<endl;
		roothist->Fill(r.X()[0],r.X()[1],height);
		i++;
	}

	//===================== Hilbert spectra analysis =======================	
	roots mid;
	int num_cont=1;
	bool same_contour=true;

	for(int i=0;i<5;i++){
		cout<<" num_cont ==== "<<num_cont<<endl;

		if(root[i].get_contour()>=0) continue;
		root[i].set_contour(num_cont-1);

		for(int j=i+1;j<5;j++){
			same_contour=true; // recover
			for(int check=0;check<100;check++){
				mid.get_mid(root[i],root[j]);
				if(q3->Eval(mid.xx(),mid.yy())>height) {
					same_contour=false; // checking 
					break;
				}	
			}
			if(same_contour) root[j].set_contour(root[i].get_contour());		
			cout<<" root[j].cont()====="<<root[j].get_contour()<<endl;
		}
		if(!same_contour) num_cont+=1;
	}	

	cout<<" after HSA num_cont == "<<num_cont<<endl;

	if(num_cont==6) goto L1;

	vector<double> first_xbar,first_ybar,second_xbar,second_ybar,third_xbar,third_ybar,fourth_xbar,fourth_ybar;
	
	for(int i=0;i<5;i++) {

		cout<<" contour tag = "<<root[i].get_contour()<<endl;
	
		if(root[i].get_contour()==0) {
			first_xbar.push_back(root[i].xx());
			first_ybar.push_back(root[i].yy());	
		}
		if(root[i].get_contour()==1) {
			second_xbar.push_back(root[i].xx());
			second_ybar.push_back(root[i].yy());	
		}
		if(root[i].get_contour()==2) {
			third_xbar.push_back(root[i].xx());
			third_ybar.push_back(root[i].yy());	
		}
		if(root[i].get_contour()==3) {
			fourth_xbar.push_back(root[i].xx());
			fourth_ybar.push_back(root[i].yy());	
		}
	
		
	}

	cout<<" first contour size ==== "<<first_xbar.size()<<endl;
	cout<<" get average of first contour ==== "<<getaverage(first_xbar)<<endl;

	std::vector<double> centroid_level;

	Double_t average_xx,average_yy;

	for(int i=0;i<num_cont;i++){
		if(i==0){
			centroid_level.push_back(q3->Eval(getaverage(first_xbar),getaverage(first_ybar)));	
			cout<<" first contour average === >"<<getaverage(first_xbar)<<"   "<<getaverage(first_ybar)<<endl;
			cout<<" first contour level = "<<q3->Eval(getaverage(first_xbar),getaverage(first_ybar))<<endl;
			average_xx = getaverage(first_xbar);
			average_yy = getaverage(first_ybar);
			cout<<" average xx = "<<average_xx<<"   "<<average_yy<<endl;
			cout<<" first contour level = "<<q3->Eval(average_xx,average_yy)<<endl;
		}	
		if(i==1){
			centroid_level.push_back(q3->Eval(getaverage(second_xbar),getaverage(second_ybar)));	
			cout<<" second contour average === >"<<getaverage(second_xbar)<<"   "<<getaverage(second_ybar)<<endl;
			cout<<" second contour level = "<<q3->Eval(getaverage(second_xbar),getaverage(second_ybar))<<endl;
		}
		if(i==2){
			centroid_level.push_back(q3->Eval(getaverage(third_xbar),getaverage(third_ybar)));	
			cout<<" third contour average === >"<<getaverage(third_xbar)<<"   "<<getaverage(third_ybar)<<endl;
			cout<<" third contour level = "<<q3->Eval(getaverage(third_xbar),getaverage(third_ybar))<<endl;
		}
		if(i==3){
			centroid_level.push_back(q3->Eval(getaverage(fourth_xbar),getaverage(fourth_ybar)));	
			cout<<" fourth contour average === >"<<getaverage(fourth_xbar)<<"   "<<getaverage(fourth_ybar)<<endl;
			cout<<" fourth contour level = "<<q3->Eval(getaverage(fourth_xbar),getaverage(fourth_ybar))<<endl;
		}
		cout<<" centroid level elements : === "<<centroid_level[i]<<endl;
	}
	Double_t level = getminimum(centroid_level);
	cout<< "level ========== "<<level<<endl;	
	
	int index = index_minimum(centroid_level);
	cout<<" index of minimum element ======"<<index<<endl;

	Double_t xxbar,yybar;
	if(index==0) {
		xxbar = getaverage(first_xbar);
		yybar = getaverage(first_ybar);
	}
	if(index==1){
		xxbar = getaverage(second_xbar);
		yybar = getaverage(second_ybar);
	}
	if(index==2){
		xxbar = getaverage(third_xbar);
		yybar = getaverage(third_ybar);
	}
	if(index==3){
		xxbar = getaverage(fourth_xbar);
		yybar = getaverage(fourth_ybar);
	}
	
	root_average_x.push_back(xxbar);
	root_average_y.push_back(yybar);

	centroid_level.clear();

	//===================== Hilbert spectaa analysis =======================

	cout<<" histogram average point ===> "<<roothist->GetMean(1)<<"   "<<roothist->GetMean(2)<<endl;
	cout<<" minimum point ====>"<<root_average_x[it+1]<<"    "<<root_average_y[it+1]<<endl;
	cout<<" xxbar and yybar =====>"<<xxbar<<"    "<<yybar<<endl;

	cout<<" flevel[it] === "<<flevel[flevel.size()-1]<<endl;

	cout<<" flevel[flevel.size()-1]) = "<<flevel[flevel.size()-1]<<endl;


	if(it==0) roothist->SetTitle(Form("f(x) = f(%.8f,%.8f) = %.8f",root_average_x[it],root_average_y[it],flevel[it]));
	else roothist->SetTitle(Form("f(x) = f(%.8f,%.8f) = %.8f",root_average_x[it],root_average_y[it],flevel[flevel.size()-1]));
	roothist->SetMarkerStyle(20);
	roothist->SetMarkerColor(it+1);
	roothist->Write();
	f3->SetName(Form("f3_%d",it));
	f3->SetContour(1,&level);
	f3->SetLineColor(it+1);
	f3->Draw("cont2");
	f3->Write();

	return level; 
}

Double_t getaverage(std::vector<double> inputvector){
	Double_t sum = 0.;
	for(int i=0;i<inputvector.size();i++) sum += inputvector[i];
	sum = sum/inputvector.size();
	return sum;
}


double getminimum(std::vector<double> inputvector){
	double min;
	min = *std::min_element(&inputvector[0],&inputvector[0]+(inputvector.size()));

	return min;
}

template<class ForwardIterator>
ForwardIterator min_element ( ForwardIterator first, ForwardIterator last )
{
if (first==last) return last;
ForwardIterator smallest = first;
while (++first!=last)
	if (*first<*smallest) smallest=first;
return smallest;

}

int index_minimum(std::vector<double> inputvector){

	int index=0;

	int size = inputvector.size();

	int n = inputvector[0];
	for(int i=1;i<size;i++){
		if(inputvector[i]<n){
			n = inputvector[i];
			index = i;
		}
	}
	return index;
}










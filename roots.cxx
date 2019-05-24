#include "roots.h"
#include "TRandom3.h"

using namespace std;

ClassImp(roots)
/*
roots::roots(){
	yy = 
}

roots::xx(){
	return x;
}
*/

roots::roots(){

	contour = -1;


}


void roots::set_values(double a,double b){
	x=a;
	y=b;
}

double roots::xx(){
	return x;

}

double roots::yy(){
	return y;
}

roots roots::mid(roots sol1, roots sol2)
{
	roots mm;
	TRandom3* mRan = new TRandom3();
	mRan->SetSeed();
	double lambda = mRan->Uniform();
	cout<<" lambda ===== "<<lambda<<endl;
	sol1.xx()+lambda*(sol2.xx()-sol1.xx());
	sol2.yy()+lambda*(sol2.yy()-sol1.yy());
	
	mm.set_values(sol1.xx()+lambda*(sol2.xx()-sol1.xx()),sol2.yy()+lambda*(sol2.yy()-sol1.yy()));	
	
	cout<<" mmm xx "<<mm.xx()<<endl;
	cout<<" mmm yy "<<mm.yy()<<endl;
	return mm;


}	

void roots::get_mid(roots sol1,roots sol2){
	TRandom3* mRan = new TRandom3();
	mRan->SetSeed();
	double lambda = mRan->Uniform();

	x = sol1.xx()+lambda*(sol2.xx()-sol1.xx());
	y = sol1.yy()+lambda*(sol2.yy()-sol1.yy());

}


int roots::scan(roots sol1, roots sol2){

	cout<<"   xx ==== "<<sol1.xx()<<endl;
	cout<<"   yy ==== "<<sol2.yy()<<endl;

	return 0;


}

void roots::set_contour(int tag){
	contour = tag; 

}

int roots::get_contour(){
	return contour;
}

roots roots::operator = (roots sol){

	return sol;
	
}


#include "roots.h"
#include "roots.cxx"
#include "contours.h"

using namespace std;

ClassImp(contours)

//template<class roots>
//contours<roots>::contours(const roots& n)
void contours::contours() {
	_capacity = 4;
	capacity = 4;
	_size = 0;
	nsize = 0;
	buffer = 0;
	cout<<" construction ================== "<<endl;
}

contours& contours::operator = (const roots & v){
	
	cout<<" 1111111111111111111111111111 "<<endl;
	delete[] buffer;
//	_size = v._size;
	cout<<" 22222222222222222222222222222 "<<endl;
	buffer = new contours [_capacity];
	cout<<" 3333333333333333333333333333"<<endl;
	for(unsigned int i = 0; i<_size;i++) buffer[i] = v.buffer[i];
	cout<<" 55555555555555555555555555555"<<endl;
	return *this;


}

contours::set_contours(const roots& n)
{

	capacity = n;
	vectorData = new <roots> [capacity];
	nsize = 0;

}

//template<class roots>
//int contours::size(contours c){

int contours::size(contours c){
	return nsize;
}

//template<class roots>
//void contours<roots>::push_back(roots& r){
void contours::push_back(roots& r){
	buffer [_size++] = r;
}

roots get_centroid(contours c){	
//	int nroots = c.size();
	int nroots = 3;
	double xbar,ybar;
	for(int i=0;i<nroots;i++){
		xbar += contours[i].xx();
		ybar += contours[i].yy();
	}
	xbar = xbar/nroots;
	ybar = ybar/nroots;

	roots sol.set_values(xbar,ybar);
	return sol;
}

contours::operator[](unsigned int index)
{

	return buffer[index];

}


/*
roots::operator = (roots sol){
	return sol;
}
*/




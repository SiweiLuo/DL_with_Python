#include "roots.h"
#include "roots.cxx"
//#include "contours.cxx"

//template<roots>
class contours{

	public:
		roots sol;
		roots centroid;
		int nsize;
		int _capacity;
		int capacity;
		roots* buffer;
		int _size;


	public:
//	   	void contours();
//		~contours();
//		void set_contours(const roots& int)
		void sort_roots(int tag);	
		void push_back(roots&);	
		int size(contours);
		roots get_centroid(contours);
	
		contours & operator = (const roots &);

//	friend roots::operator =;
};

#endif


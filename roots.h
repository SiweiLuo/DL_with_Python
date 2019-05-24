class roots {

	public: 
		Double_t x;
		Double_t y; 
//	private:
		int contour = -1;
	
	public:
		void set_values(double,double);	
		void get_mid(roots,roots);
		int scan(roots, roots);
		Double_t xx();
		Double_t yy();
		roots mid(roots,roots);
		
		void set_contour(int);
		int get_contour();

		roots operator = (roots);

}solutions;






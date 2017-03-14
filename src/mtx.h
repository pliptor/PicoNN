// Copyright (C) Oscar Takeshita 2017
// A minimum matrix manimulation library to run the PicoNN 
// This library was not written with efficiency as a goal
#ifndef _MTX_H_
#define _MTX_H_
#include<algorithm>
#include<cassert>
#include<vector>
#ifndef _unused
#define _unused(x) ((void)(x))
#endif
class mtx {  
	private: 
		int rdim;
		int cdim;
		std::vector<field> v;
	public:
		mtx() {;};
		mtx(int r, int c) { assert(r>0 && c>0); rdim = r; cdim = c; v.resize(r*c, static_cast<field>(0.)); };
		mtx(mtx &a) { copy(a); };
		void ld(field value) { std::fill(v.begin(), v.end(), value); }; 
		void  set(int i,        field value) { assert(i>=0 && i<rdim*cdim); v[i] =  value; } // linear access
		void  set(int r, int c, field value) { assert(r>=0 && r<rdim); assert(c>=0 && c<cdim); v[c + r*cdim] =  value; }
		void  add(int r, int c, field value) { assert(r>=0 && r<rdim); assert(c>=0 && c<cdim); v[c + r*cdim] += value; }
		field get(int r, int c = 0)          { assert(r>=0 && r<rdim); assert(c>=0 && c<cdim); return v[c + r*cdim]; }
		void init(int r, int c) { assert(r>0 && c>0); v.clear(); rdim = r; cdim = c; v.resize(r*c, static_cast<field>(0.)); };
		void print_size() { printf("(%d, %d)\n", rdim, cdim);};
		void print(int prec) {
			for( int i = 0; i< rdim; i++) {
				for( int j = 0; j< cdim; j++) 
					fprintf(stdout, "%.*f ",prec, v[j + i*cdim]);
				fprintf(stdout,"\n");
			}				
		}

		// layer functions
		void clr() { std::fill(v.begin(), v.end(), static_cast<field>(0.)); }; 
		void add_all(field value)  { for (auto &a : v) { a += value;}; };
		void mlt_all(field value)  { for (auto &a : v) { a *= value;}; };
		void div_all(field value)  { for (auto &a : v) { a /= value;}; };
		void ReLU()                { for (auto &a : v) { a = std::max(static_cast<field>(0.), a); };};
		void exp()                 { for (auto &a : v) { a = std::exp(a); };};
		void rnd(field k, rand_field &rd)  { for (auto &a : v) { a = k*rd.randn(); };};
		std::vector<field> &vec()  { return v; };
		std::vector<field> *vecp() { return &v; };

		void dReLU(mtx &b)               { 
			auto *bv = b.vecp();
			std::transform(v.begin(), v.end(), bv->begin(), v.begin(), [](field &l, field const &r) { return r<=0 ? static_cast<field>(0.) : l ;}); 
		}

		// multiply matrix a by b and add an optional row vector k to each row in axb
		// set ta/tb true if ta and tb are to be transposed
		void  mult(mtx&a, bool ta, mtx& b, bool tb, void *k = NULL) {
			int rdim = ta ? a.cdim   : a.rdim;
			int cdim = tb ? b.rdim   : b.cdim;
			int commona = ta ? a.rdim : a.cdim;
			int commonb = tb ? b.cdim : b.rdim;
			_unused(commonb);
			assert(commona == commonb); // A and B can be multiplied 
			assert(this->rdim == rdim && this->cdim == cdim);     // *this matrix size is compatible with product 

			for(int i = 0; i< rdim; i++) {
				for(int j = 0; j< cdim; j++) {
					this->set(i, j, static_cast<field>(0.));
					for(int z = 0; z< commona; z++ ) 
						if(!ta && !tb) 
							this->add(i, j, a.get(i , z) * b.get(z, j));
						else if(!ta && tb) 
							this->add(i, j, a.get(i , z) * b.get(j, z));
						else if(ta && !tb) 
							this->add(i, j, a.get(z , i) * b.get(z, j));
						else if(ta && tb) 
							this->add(i, j, a.get(z , i) * b.get(j, z));

					if(k!=NULL)
						this->add(i, j, static_cast<mtx*>(k)->get(j));
				}

			}
		}
		void marg(mtx &b, bool down = true) { //  down true: collumns collapse to form marginal sum ; down false: rows collapse
			assert(this->cdim == 1 || this->rdim == 1);
			assert(std::max(this->cdim, this->rdim) == (down ? b.cdim : b.rdim)); // if collapsing down, resulting vector has length cdim
			this->clr();
			int r = b.rdim;
			int c = b.cdim;
			if(!down)
				std::swap(r,c);
			for(int i = 0; i< b.cdim; i++)
				for(int j = 0; j< b.rdim; j++)
					this->add(i , 0, b.get(j,i));

		} 
		void linear_add(mtx &b, field k = 1.) { // generalize later to add another matrix if needed
			auto *bv = b.vecp();
			std::transform(v.begin(), v.end(), bv->begin(), v.begin(), [k](field &l, field &r ) { return l + k*r;});
		} 
		field sum() { return std::accumulate(v.begin(), v.end(), static_cast<field>(0.));};
		field L2() { return std::inner_product(v.begin(), v.end(), v.begin(), static_cast<field>(0.));};
		void copy(mtx &b) {
			init(b.rdim, b.cdim);
			this-> v = b.vec();
		}
};
#endif

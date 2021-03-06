// Copyright (C) Oscar Takeshita 2017
// data class for PicoNN
#ifndef _PNNDATA_H_
#define _PNNDATA_H_
#include "mtx.h"
class t_data { // base class for data 
	protected:
		int K;       // number of classes
		int D;       // number of input dimensions
		int U;       // number of data points 
	public:
		mtx X;       // data
		mtx Y;       // labels
		virtual void build() {}                      // populates X and Y
		virtual void print_train(bool svm = true) {} // svm ? (print data in libsvm format) : (print in plain format) 
		int get_U() { return U; };
		int get_K() { return K; };
		int get_D() { return D; };
		virtual void print_train() {} // print data in libsvm format 
};

// This class generates a data model.
// It is a collection of points in D dimensions. Each point belongs to one of K classes.
// There is a total of N*K points that get configured in a spiral shape.
// The configuration is such that grouping by class cannot be done by drawing a
// few straight lines. It needs fancier boundaries and the neural network in the
// example can handle the generation of such boundaries.
//    The spiral class is a translation from Python to C++ available in
//    http://cs231n.github.io/neural-networks-case-study/
class spiral : public t_data {
	private:
		rand_field *rd;
		void build() {
			X.init(U, D);
			Y.init(U, 1);
			const field arm_step = 4.;
			for (int label = 0; label<K ; label++)  {
				// build spiral arms
				field  r = 0.;               // radius
				field dr = 1./(U/K-1);
				field  t = arm_step*label;  // angle
				field dt = arm_step/(U/K-1);
				for (int i = 0; i<U/K; i++) {
					field phase_noise = 0.2*rd->randn();
					X.set(i + label*U/K, 0,  r * static_cast<field>(std::sin(t + phase_noise)));  // first  dimension
					X.set(i + label*U/K, 1,  r * static_cast<field>(std::cos(t + phase_noise)));  // second dimension
					Y.set(i + label*U/K, 0,  static_cast<field>(label));
					r += dr;
					t += dt;
				}
			}
		};
	public:
		spiral(int N, int K, rand_field &rd) {
			assert(K>0 && N>0 && D>0);
			this->U = N*K;
			this->D = 2;
			this->K = K;
			this->rd = &rd;
			build();
		}

		void print_train(bool svm = true) { // print data in libsvm format (label 1:first_dimension 2:second_dimension)
			for(int i = 0; i<U; i++)
				if(svm)	
					std::printf("%d 1:%.8f 2:%.8f\n", static_cast<int>(Y.get(i)), X.get(i, 0), X.get(i, 1));
				else
					std::printf("%d   %.8f   %.8f\n", static_cast<int>(Y.get(i)), X.get(i, 0), X.get(i, 1));
		};
};
#endif

// Copyright (C) Oscar Takeshita 2017
// This code is a translation of the 2-layer neural network Python code in 
//    http://cs231n.github.io/neural-networks-case-study/
//    Andrej Karpathy
//    to C++ with some modifications. 
//
// The machine learning course webpage and the code are quite instructive and I
// decided to write this translation for my own experiments 
// and for those wanting to play with a C++ version. 

#include<algorithm>
#include<cassert>
#include<vector>
#include<fstream>
#include<sstream>
#include<iostream>

typedef double field; // field of real numbers

// RNG generator
static std::mt19937 generator;     
static std::normal_distribution<field> dist; 
class rand_field {
	public:
		field randn() {
			return dist(generator);
		}
};

#include "mtx.h"
#include "pnndata.h"
class network {
	private:
		int D;           // input data dimension
		int K;           // number of classes
		int h;           // number of neurons in hidden layer
		int N;
		int U;           // batch size

		rand_field *rd;  // normal gaussian random variable
		t_data *tdt;     // data class

		field step_size; // gradient descent step size
		field reg;       // regularization strength
		field loss;
		field data_loss;
		field reg_loss;

		void build() {
			fprintf(stderr, "Network parameters: Nodes per class  N = %d, Dimension D = %d   Classes K = %d   Hidden nodes h = %d\n",N, D, K, h);
			W1.init(D, h);      
			b1.init(h, 1);
			W2.init(h, K);
			b2.init(K, 1);

			dW1.init(D, h);
			db1.init(h, 1);
			dW2.init(h, K);
			db2.init(K, 1);

			Hidden.init(U, h); 
			dHidden.init(U, h);    
			Scores.init(U, K);

			Probs.init(U, K);
			step_size = static_cast<field>(1.);
			reg =       static_cast<field>(1e-3);
			loss =      static_cast<field>(0.);
			data_loss = static_cast<field>(0.);
			reg_loss =  static_cast<field>(0.);
		}

	public:
		mtx W1, b1;
		mtx W2, b2;

		mtx dW1, db1;
		mtx dW2, db2;

		// hidden layer output
		mtx Hidden;
		mtx dHidden;

		mtx Scores;
		mtx Probs;
		void set_loss(field l)      { loss      = l; }; 
		void set_data_loss(field l) { data_loss = l; }; 
		void set_reg_loss(field l)  { reg_loss  = l; }; 

		field get_loss()      { return loss      ; }; 
		field get_data_loss() { return data_loss ; }; 
		field get_reg_loss()  { return reg_loss  ; }; 

		// gradient descent step sizes 
		// normally between 1. and smaller values
		void set_step_size(field step_size) {
			this->step_size = step_size;
		}

		// regularization parameter
		void set_reg(field reg) {
			this->reg = reg;
		}
		void network_state(int i) {
			fprintf(stderr, "**** network state dump ****\n\n");
			fprintf(stderr, "%d W2:\n",i);      W2.print(10);
			fprintf(stderr, "%d b2:\n",i);      b2.print(10);
			fprintf(stderr, "%d scores:\n",i);  Scores.print(8);
			fprintf(stderr, "%d probs:\n",i);   Probs.print(8);
			fprintf(stderr, "%d loss(%.10f) = data loss(%.10f) + reg loss(%.10f):\n",i, get_loss(), get_data_loss(), get_reg_loss()); 
			fprintf(stderr, "%d hidden:\n",i);  Hidden.print(20);
			fprintf(stderr, "%d dW2:\n",i);     dW2.print(8);
			fprintf(stderr, "%d db2:\n",i);     db2.print(8);
			fprintf(stderr, "%d dhidden:\n",i); dHidden.print(20);
			fprintf(stderr, "%d dW1:\n",i);     dW1.print(20);
			fprintf(stderr, "%d db:\n",i);      db1.print(20);
			fprintf(stderr, "%d W1:\n",i);      W1.print(8);
			fprintf(stderr, "%d b:\n",i);       b1.print(8);
			fprintf(stderr, "**** dump completed ****\n\n");
		}

		// function to be called to set the initial state of the network
		void initialize_net(field W1std, field b1const, field W2std, field b2const) { 
			W1.rnd(W1std,  *rd); // weight is normal distributed with std deviation W1std
			b1.ld(b1const);      // bias initialized with constant b1const  
			W2.rnd(W2std,  *rd); // weight is normal distributed with std deviation W2std
			b2.ld(b2const);      // bias initialized with constant b2const
		}

		// forward pass routines
		void evaluate_class_scores(mtx input) { 
			Hidden.mult(input, false, W1, false, static_cast<void*>(&b1));  // Hidden = X * W1 + b1
			Hidden.ReLU();   // Non-linearity 
			Scores.mult(Hidden, false, W2, false, static_cast<void*>(&b2)); // Scores = Hidden * W2 + b2
		}

		void compute_class_probabilities() { 
			mtx Expscores(Scores);
			Expscores.exp();
			auto expscores = Expscores.vec();
			std::vector<field> sum(U);
			for(int i = 0; i<U; i++)
				sum[i] = std::accumulate(expscores.begin() + i*K, expscores.begin() + (i+1)*K, static_cast<field>(0.));
			for(int i = 0; i<U; i++)
				std::transform(expscores.begin() + i*K, expscores.begin() + (i+1)*K,
						Probs.vecp()->begin() + i*K, [i, sum](field &p) { return p/sum[i];} );
		}

		// Note that none of these computations affect the optimization
		// algorithm (loss is not used either directly or to control parameters)
		// It is only used for monitoring at the moment
		void compute_loss() { 
			double dloss = 0.;
			for(int i = 0; i<U; i++)
				dloss -= std::log(Probs.get(i, tdt->Y.get(i)));
			set_data_loss(static_cast<field>(dloss)/(U));
			set_reg_loss(static_cast<field>(0.5 * reg) * (W1.L2() + W2.L2()));
			set_loss(get_data_loss() + get_reg_loss());
		}

		void compute_gradient_on_scores() { 
			for (int i = 0; i<U ; i++)
				Probs.add(i, tdt->Y.get(i), -1.);
			Probs.div_all(U);
		}

		void compute_dW2db2() { //
			dW2.mult(Hidden, true, Probs, false); // dW2 = t(Hidden) * dScores
			db2.marg(Probs);
		}

		void compute_dhidden() {
			dHidden.mult(Probs, false, W2, true); // dHidden = dScores * t(W2)
		}

		void compute_dReLU() {
			dHidden.dReLU(Hidden);
		}

		void compute_dWdb() {
			dW1.mult(tdt->X, true, dHidden, false);
			db1.marg(dHidden, true); 
		}

		// backprop gradient into W2 and b2
		void back_propagate() {
			compute_dW2db2();
			compute_dhidden();
			compute_dReLU();
			compute_dWdb();
		}

		// add regularization gradient constribution
		void add_regularization() {
			dW1.linear_add( W1, reg);      // dW1  += reg * W1
			dW2.linear_add( W2, reg);      // dW2  += reg * W2
		}

		// descend the cost topology by applying adding a negative scaled value of the gradient
		void descend() {
			W1.linear_add(dW1, -step_size); // W1 += -stepsize * dW1
			b1.linear_add(db1, -step_size); // b1 += -stepsize * db1
			W2.linear_add(dW2, -step_size); // W2 += -stepsize * dW2
			b2.linear_add(db2, -step_size); // b2 += -stepsize * db2
		}

		// THE GRADIENT DESCENT. It optimizes the cost function.
		void gradient_descent(unsigned int iterations) {
			for(unsigned int i = 0; i<iterations; i++) {
				evaluate_class_scores(tdt->X);
				compute_class_probabilities();
				compute_loss();
				compute_gradient_on_scores();
				back_propagate();
				add_regularization();
				descend();
				if(i%1000 == 0) {
					fprintf(stdout, "iteration %6d: loss %f data_loss %f reg_loss %f     ", i, get_loss(), get_data_loss(), get_reg_loss());
					accuracy();
					//network_state(i); // uncomment for full dump of network state
				}
			}
		}

		// network constructor 
		// It builds network topology using 
		// model parameters
		network(int h, t_data &tdt, rand_field &rd) {
			this->h = h;
			this->tdt = &tdt;
			this->N = tdt.get_N();
			this->K = tdt.get_K();
			this->D = tdt.get_D();
			this->rd = &rd;
			this->U = N*K;
			build();
		}


		// reports training accuracy
		void accuracy() {
			int hit = 0;
			for(int i = 0; i< U; i++) {
				field best_score = -10000;
				int best_index = -1;
				for(int j = 0; j<K; j++) {
					if(Scores.get(i,j)>best_score) {
						best_score = Scores.get(i,j);
						best_index = j;
					}
				}
				assert(best_index != -1);
				if(best_index == tdt->Y.get(i))
					hit++;
			}
			std::fprintf(stdout, "training accuracy %.2f%%\n", static_cast<field>(hit)*100/(U));
		}

		// predicts on model. The input_file is contains
		// the data to be classified in csv format.
		// The output is written in csv format as well
		void predict(std::string input_file, std::string output_file) {
			mtx A;
			double x, y;
			A.init(1,2);
			std::string line;
			std::ifstream in (input_file.c_str());
			std::ofstream out (output_file.c_str());
			if(!out.is_open()) {
				std::cerr << "problem opening " << output_file << std::endl;
				return;
			}
			else {
				out << "Label,X,Y\n";
			}
			if(in.is_open()) {
				getline(in, line); // discard header
				while(getline(in, line, ',')) {
					std::istringstream(line) >> x; A.set(0, 0, x);
					getline(in, line);
					std::istringstream(line) >> y; A.set(0, 1, y);
					evaluate_class_scores(A);
					float best_score = Scores.get(0, 0);
					int best_label = 0;
					for(int j = 1; j<K ;j++) {
						if(Scores.get(0, j) > best_score) {
							best_label = j;
							best_score = Scores.get(0, j);
						}
					}				
					out << best_label << "," << x << "," << y << std::endl;
				}
				in.close();
				out.close();
			}
			else {
				std::cerr  << "problem opening " << input_file << std::endl;
				exit(-1);
			}
		}
};

int main(int argc, char *argv[]) {
	// normal distributed RV
	if(argc == 2)
		generator.seed(atoi(argv[1]));
	else
		generator.seed(1);
	rand_field rd;

	// Data model spiral : U points inside a 2 x 2 square 
	// Points belong to K different classes and the configuration is such that
	// there are no linear boundaries that separates each class. In other words,
	// they can't be grouped by drawing a few straight lines. See figure...

	int N = 100;                 // number of points per class
	int K = 3;                   // number of classes
	spiral tdt(N, K, rd);
	//tdt.print_train();           // prints the train data in libsvm format 

	// Setup a 2-layer neural network 
	// plus the number of hidden nodes
	int h = 100;                 // hidden nodes 
	network nn(h, tdt, rd);

	// Initial conditions for the network
	field W1, b1, W2, b2;         
	W1 = 0.01;
	b1 = 0.00;
	W2 = 0.01;
	b2 = 0.00;
	nn.initialize_net(W1, b1, W2, b2);

	// Optimization parameters
	int iter = 10000;            // number of iterations
	nn.set_reg(1e-3);            // regularization
	nn.set_step_size(1.);        // gradient descent step size.

	// Starts training/optimization
	nn.gradient_descent(iter);

	// Reports training accuracy
	nn.accuracy();               

	// Read test cordinates and write predictions
	//nn.predict("../extras/test.csv","../extras/prediction.csv");

	return 0;
}


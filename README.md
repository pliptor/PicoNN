# PicoNN

## A small 2-layer neural network in C++

[PicoNN](https://github.com/pliptor/PicoNN)

* It's a fully functional, self-contained, C++ 2-layer neural network testing and study platform.
* It has less than 500 lines of code but has zero dependencies. No Theano, no Python, no CUDA, no linear algebra libraries.
* It is not meant for a full blown application platform. It is meant for those that
are trying to understand basic details of NN and want to have some fun and look at the individual components at the lower levels in simple C++ code. If you are looking for a comprehesive platform you may want to try [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn). 
* Concepts you may learn with this code:
	- Forward propagation
	- Back propagation
	- Gradient descent
	- Regularization
	- ReLU activation
	- Metrics to track convergence
* The code is a translation of the Python code available in the excellent freely available neural network [cs231n course](http://cs231n.github.io/neural-networks-case-study/) by Andrej Karpathy. You are encouraged to check his webpage to understand all the theory in great detail. (You may most likely learn with his Python code instead as it is much easier to follow).

## What does the code do?

It segments a group of $K\times N$ points, where $K=3$ and $N=100$ using a 2-layer neural network with 100 nodes in the hidden layer.

![segmented spiral](http://cs231n.github.io/assets/eg/spiral_net.png) This is a plot of the resulting segmentation found in the cs231n webpage.

## HowTo

This code was developed in Linux (Ubuntu 16.04) but it should be easy to build in any platform with a C++11 compiler since it has no dependencies and just one cpp file and two header files.

* In Linux, type `make` to build the executable `piconn`, which will be found in the `build` folder.
* To run the the executable type `./piconn`

The output should be 

~~~

~~~

The last line shows that the segmentation was performed with 99% accuracy. The result will not perfectly match the original Python code. This is because of differences in random number generation.



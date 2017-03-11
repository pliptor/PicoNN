# PicoNN

## A small 2-layer neural network in C++

|Unix |
|-------|
|       |

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

It segments a group of KxN points, where K=3 and N=100 using a 2-layer neural network with 100 nodes in the hidden layer.

![segmented spiral](http://cs231n.github.io/assets/eg/spiral_net.png) This is a plot of the resulting segmentation found in the cs231n webpage.

## HowTo

This code was developed in Linux (Ubuntu 16.04) but it should be easy to build in any platform with a C++11 compiler since it has no dependencies and just one cpp file and two header files.

* In Linux, type `make` to build the executable `piconn`, which will be found in the `build` folder.
* To run the the executable type `./piconn`

The output should be 

~~~
iteration      0: loss 1.098716 data_loss 1.098692 reg_loss 0.000023     training accuracy 27.67%
iteration   1000: loss 0.286646 data_loss 0.179557 reg_loss 0.107089     training accuracy 92.67%
iteration   2000: loss 0.273426 data_loss 0.152912 reg_loss 0.120514     training accuracy 96.33%
iteration   3000: loss 0.251950 data_loss 0.125369 reg_loss 0.126581     training accuracy 98.00%
iteration   4000: loss 0.246508 data_loss 0.116324 reg_loss 0.130184     training accuracy 98.33%
iteration   5000: loss 0.245213 data_loss 0.113930 reg_loss 0.131283     training accuracy 98.67%
iteration   6000: loss 0.243728 data_loss 0.111983 reg_loss 0.131746     training accuracy 99.00%
iteration   7000: loss 0.242398 data_loss 0.110968 reg_loss 0.131430     training accuracy 99.00%
iteration   8000: loss 0.242011 data_loss 0.110887 reg_loss 0.131124     training accuracy 99.00%
iteration   9000: loss 0.241819 data_loss 0.110861 reg_loss 0.130958     training accuracy 99.00%
training accuracy 99.00%
~~~

The last line shows that the segmentation was performed with 99% accuracy. The result will not perfectly match the original Python code. This is because of differences in random number generation.



![neuron](https://github.com/pliptor/PicoNN/blob/master/extras/neuron_cropped.png)

# PicoNN

## A small 2-layer neural network in C++

|Unix |
|-------|
|![travis](https://travis-ci.org/pliptor/PicoNN.svg?branch=master)|

[PicoNN](https://github.com/pliptor/PicoNN)

* It's a fully functional, self-contained,  neural network testing and studying platform.
* It has **less than 500 lines of code** and yet no dependencies. If you understand every line of the code, **you might have understood many of the principles of modern neural network**. It does assume you know a little about linear algebra such as matrix multiplications.
* It is not meant for building a full blown application. You may want to try [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) for a comprehensive platform. 
* Concepts you may learn with this code:
    - *Forward propagation*
    - *Back propagation*
    - *Gradient descent*
    - *Regularization*
    - *ReLU activation*
    - *Metrics to track convergence*
    - *Effects of floating point precision*
    - *Effects of weight and bias initialization*
* The code is a translation of the Python code available in the excellent and freely available neural network [cs231n course](http://cs231n.github.io/neural-networks-case-study/) by Andrej Karpathy. You are encouraged to check his web-page that has all the theory in great detail. Also, you may try his original Python code instead.

## What does the code do?

It segments a group of KxN=300 points, where K=3 and N=100, using a 2-layer neural network with 100 nodes in the hidden layer.

Data | Segmented Data
:------:|:--------:
![Data](https://github.com/pliptor/PicoNN/blob/master/extras/input.png) |![Segmented Data](https://github.com/pliptor/PicoNN/blob/master/extras/output.png)

## How To

This code was developed in Linux (Ubuntu 16.04.1LTS) but it should be easy to build it in any platform with a C++11 compiler. It has no dependencies and just one cpp file and two header files.

* In Linux, type `make` to build the executable `piconn`, which will be then found in the `build` folder.
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



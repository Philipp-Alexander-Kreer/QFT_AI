# QFT AI

## Introduction

In this repository, I explore the connection between Quantum Field Theory and Artificial Intelligence. The main goal is to understand how the principles of Quantum Field Theory can be applied to the field of 
Artificial Intelligence by reproducing the results of the paper [Neural Networks and Quantum Field Theory](https://arxiv.org/abs/2008.08601). 
In this paper, the authors compute correlation functions of neural networks using an Effective Field Theory description. 
The Effective Field Theory is defined by an expansion in the ration $n=\frac{L}{w}$, where $w$ is the width of the network and $L$ is the number of layers.
The n-pt functions of the neurel network at initialization are than given as a sum of Feynman diagrams. I reproduce the experiments for the ReLU activation function is mostly used in practical applications.

## Structure of Program

The main program consist of four functions:

1. generate_data: This function initializes neural networks as described in the paper and computes the correspondig 2-pt, 4-pt and 6-pt functions.
2. experiment_1:  This function reproduces the findings of the first experiment summarized in Fig. 2 of the paper.
   It verifies that the 2-pt function of the free effective field theory equals the 2-pt function of the neural network at initialization.
3. experiment_2a: This function reproduces the findings of the second experiment summarized in Fig. 5 (left) of the paper. It measures the interaction strength entering the interacting effective field theory.
4. experiment_2b: This function reproduces the findings of the second experiment summarized in Fig. 5 (right) of the paper. Using the coupling constant obtained from experiment_2a, 
we predict the 6-pt function of the neural network at initialization.

## Installation

The required standard Python libraries are: numpy, matplotlib, scipy, itertools, and torch.

Git clone the repository and run the main program main.py. This version is shipped with data for various network sizes such that the experiments can be reproduced without data generation.
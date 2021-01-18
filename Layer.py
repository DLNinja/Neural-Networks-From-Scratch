import numpy as np
from ActivationFunctions import *
from LossFunctions import *

"""
    With this class we create the layers that will go in the neural network
    Every layer will have their own set of biases, and a set of weights between them and the prior layer
    This set of weights is different for every pair of layers, it has random values in the start, but this values
    will be changed after each backpropagation process.
    This is not the final form, I'll add much more in the future
"""

"""
    Our Dense class creates the layer, taking as parameters the folowing:
        - layerSize: the size of the actual layer
        - activation: this specifies the activation function we'll use for this layer
        - weightBounds: the interval in which all weights will be initialized
"""

class Dense:
    def __init__(self, layerSize, activation="sigmoid", weightBounds=(-1, 1)):
        self.length = layerSize
        self.bounds = weightBounds

        # self.weights = np.random.uniform(weightBounds[0], weightBounds[1], (layerSize, inputSize))
        # self.biases = np.zeros((layerSize, 1))
        self.activation = activation
        self.derivative = activation
        if self.activation == "relu":
            self.derivative = ReLU_prime
            self.activation = ReLU
        elif self.activation == "tanh":
            self.activation = tanh
            self.derivative = tanh_prime
        elif self.activation == "softmax":
            self.activation = softmax
            self.derivative = L1
        else:
            self.activation = sigmoid
            self.derivative = sigmoid_prime

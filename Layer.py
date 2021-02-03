import numpy as np
from ActivationFunctions import *
from LossFunctions import *

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
            self.derivative = lambda x: 1
        else:
            self.activation = sigmoid
            self.derivative = sigmoid_prime

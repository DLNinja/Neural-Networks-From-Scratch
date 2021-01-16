import numpy as np
from ActivationFunctions import *
np.random.seed(0)
"""
    With this class we create the layers that will go in the neural network
    Every layer will have their own set of biases, and a set of weights between them and the prior layer
    This set of weights is different for every pair of layers, it has random values in the start, but this values
    will be changed after each backpropagation process.
    This is not the final form, I'll add much more in the future
"""

"""
    Our Dense class creates the layer, taking as parameters the folowing:
        - inputSize: the size of the array which is the previous layer
        - layerSize: the size of the actual layer
        - activation: this specifies the activation function we'll use for this layer
        - weightBounds: the interval in which all weights will be initialized
"""

class Dense:
    def __init__(self, inputSize, layerSize, activation="sigmoid", weightBounds=(-1, 1)):
        self.x = inputSize
        self.y = layerSize
        self.weights = np.random.uniform(weightBounds[0], weightBounds[1], (layerSize, inputSize))
        self.biases = np.zeros((layerSize, 1))
        self.activation = activation
        self.derivative = activation

    def forward(self, inputLayer):  # weights and input layer are multiplied and than the activation function is applied
        self.z = np.dot(self.weights, inputLayer) + self.biases # the layer before applying the activation function
        if self.activation == "ReLU":
            self.derivative = ReLU_prime
            self.output = np.array([ReLU(x) for x in self.z])
        elif self.activation == "tanh":
            self.output = np.array([tanh(x) for x in self.z])
            self.derivative = tanh_prime
        elif self.activation == "softmax":
            self.output = softmax(self.z)
        else:
            self.output = np.array([sigmoid(x) for x in self.z])
            self.derivative = sigmoid_prime

import numpy as np
from ActivationFunctions import ReLU
from ActivationFunctions import sigmoid
from ActivationFunctions import tanh
from ActivationFunctions import softmax
np.random.seed(0)
"""
    With this class we create the layers that will go in the neural network
    Every layer will have their own bias, and a set of weights between them and the prior layer
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
        self.weights = np.random.uniform(weightBounds[0], weightBounds[1], (inputSize, layerSize))
        self.bias = 1
        self.acFunction = activation

    def forward(self, inputLayer):
        self.output = np.dot(inputLayer, self.weights)
        if self.acFunction == "ReLU":
            activation = ReLU
        elif self.acFunction == "sigmoid":
            activation = sigmoid
        elif self.acFunction == "tanh":
            activation = tanh
        elif self.acFunction == "softmax":
            activation = softmax
        self.output = activation(self.output)

"""   Testing section   """

i = [2, 3, 2.5]
l1 = Dense(3, 4, "ReLU")
l1.forward(i)
l2 = Dense(4, 2, "softmax")
l2.forward(l1.output)
print(l1.output)
print(l2.output)
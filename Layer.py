import numpy as np

"""
    With this class we create the layers that will go in the neural network
    Every layer will have their own bias, and a set of weights between them and the prior layer
    This set of weights is different for every pair of layers, it has random values in the start, but this values
    will be changed after each backpropagation process.
    This is not the final form, I'll add much more in the future
"""

class Dense:
    def __init__(self, inputSize, layerSize):
        self.x = inputSize
        self.y = layerSize
        self.weights = np.random.randn(inputSize, layerSize)
        self.bias = 1

    def forward(self, inputLayer):
        self.output = np.dot(inputLayer, self.weights) + self.bias

import numpy as np
from Layer import Dense

"""
    This is the Neural Network class itself.
    For now, the network knows how to add new layers and feed forward the information.
    Next, I'll add the backprop function which will help improving the accuracy by tweaking the weights.
    After that, I'll add other things like training with more sets of inputs, a summary of the net, batches etc.
"""

class NeuralNetworkModel:
    def __init__(self, inputSize, outputSize):
        self.x = inputSize
        self.y = outputSize
        self.layers = []  # this stores the layers

    def add(self, newLayer):  # like the name suggests, it adds layers to the net
        self.layers.append(newLayer)

    def feedforward(self, input):  # The calculations are done for each of the layers
        priorLayer = input
        for newLayer in self.layers:
            newLayer.forward(priorLayer)
            priorLayer = newLayer.output
        return newLayer.output


""" 
    Testing section
    So putting all the stuff from the testing section in the Layer file, we can see that this automated version spits
    the exact same values, yay
"""

t = NeuralNetworkModel(3, 2)
t.add(Dense(3, 4, "ReLU"))
t.add(Dense(4, 2, "softmax"))
y = t.feedforward([2, 3, 2.5])
print(y)
import numpy as np
from Layer import Dense
from LossFunctions import *
from ActivationFunctions import *

np.random.seed(0)

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
        self.sizes = [inputSize]
        self.layers = [np.zeros((1, inputSize))]  # this stores the layers
        self.weights = []
        self.biases = []

    def add(self, newLayer):  # like the name suggests, it adds layers to the net
        self.layers.append(newLayer)
        self.sizes.append(newLayer.length)
        self.weights.append(np.random.randn(self.sizes[-1], self.sizes[-2]))
        self.biases.append(np.zeros((self.sizes[-1], 1)))

    def feedforward(self, input):  # The calculations are done for each of the layers
        x = np.transpose([input])
        for i in range(1, len(self.layers)):
            x = self.layers[i].activation(np.dot(self.weights[i-1], x) + self.biases[i-1])
        return x

    def backprop(self, x, y):  # Ah, yess, the most important step ( which involves a lot of math)
        # this two lists will represent the changes in weights and biases
        # with which will update the network to give better results
        b_change = [np.zeros(b.shape) for b in self.biases]
        w_change = [np.zeros(w.shape) for w in self.weights]
        # we apply feedforward on the layers, and then start the backprop part
        outputs = [x]  # this array takes the layers after activation function
        layer = x
        zs = []  # this array takes the layers before activation function
        for (l, w, b) in zip(self.layers[1:], self.weights, self.biases):
            z = np.dot(w, layer) + b
            layer = l.activation(z)
            zs.append(z)
            outputs.append(layer)
        delta = np.array(layer - y)  # * ReLU(self.layers[-1].z)
        b_change[-1] = delta
        w_change[-1] = np.dot(delta, outputs[-2].T)
        for l in range(2, len(self.layers)):
            prime = ReLU_prime(zs[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * prime
            b_change[-l] = delta
            w_change[-l] = np.dot(delta, outputs[-l-1].T)
        return w_change, b_change

    def update_batch(self, xt, yt):
        b_change = [np.zeros(b.shape) for b in self.biases]
        w_change = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in zip(xt, yt):
            x = np.transpose([x])
            y = np.transpose([y])
            dw, db = self.backprop(x, y)
            b_change += db
            w_change += dw
        return w_change, b_change

    # this is a simplified version of the train method, for now it works with just 1 input
    def train(self, xt, yt, it, alpha):
        acc = []
        for i in range(it):
            dw, db = self.update_batch(xt, yt)
            self.weights = [w - alpha * ndw for (w, ndw) in zip(self.weights, dw)]
            self.biases = [b - alpha * ndb for (b, ndb) in zip(self.biases, db)]
            acc.append(L1(self.feedforward(np.transpose(xt[0])), np.transpose(yt[0])))
        return acc

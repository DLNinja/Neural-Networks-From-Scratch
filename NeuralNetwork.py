import numpy as np
from Layer import Dense
from LossFunctions import *
from ActivationFunctions import *

np.random.seed(0)

"""
    This is the Neural Network class itself.
    
    It looks a lot like the one from Michel Nielsen's book and that's because his method is just the best
    Tried a lot of methods but this was on another level
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
        delta = np.array(layer - y) * self.layers[-1].activation(zs[-1])
        b_change[-1] = delta
        w_change[-1] = np.dot(delta, outputs[-2].T)
        for l in range(2, len(self.layers)):
            prime = self.layers[-l].derivative(zs[-l])
            delta = np.dot(self.weights[-l+1].T, delta) * prime
            b_change[-l] = delta
            w_change[-l] = np.dot(delta, outputs[-l-1].T)
        return w_change, b_change

    def update_batch(self, xt, yt, alpha):
        b_change = [np.zeros(b.shape) for b in self.biases]
        w_change = [np.zeros(w.shape) for w in self.weights]
        for (a, b) in zip(xt, yt):
            x = np.transpose([a])
            y = b
            dw, db = self.backprop(x, y)
            b_change = [bc+dbc for bc, dbc in zip(b_change, db)]
            w_change = [wc+dwc for wc, dwc in zip(w_change, dw)]
        self.weights = [w - (alpha / len(xt)) * ndw for (w, ndw) in zip(self.weights, w_change)]
        self.biases = [b - (alpha / len(xt)) * ndb for (b, ndb) in zip(self.biases, b_change)]

    # this is a simplified version of the train method, for now it works with just 1 input
    def train(self, xt, yt, it, alpha, batch_size, x_test, y_test):
        for i in range(it):
            for k in range(0, len(xt), batch_size):  # updating in batches -  needs to be changed
                self.update_batch(xt[k:k+batch_size], yt[k:k+batch_size], alpha)
            result = 0
            for (x, y) in zip(x_test, y_test):
                output = self.feedforward(x)
                result += int(np.argmax(output) == y)
            print("Epoch {0}: {1} / {2}".format(i + 1, result, len(y_test)))

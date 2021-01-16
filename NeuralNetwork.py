import numpy as np
from Layer import Dense
from LossFunctions import *
from ActivationFunctions import *
import matplotlib.pyplot as plt

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

    def backprop(self, x, y): # Ah, yess, the most important step ( which involves a lot of math)
        # this two lists will represent the changes in weights and biases
        # with which will update the network to give better results
        b_change = [np.zeros(b.biases.shape) for b in self.layers]
        w_change = [np.zeros(w.weights.shape) for w in self.layers]
        # we apply feedforward on the layers, and then start the backprop part
        delta = np.array(self.feedforward(x) - y) * sigmoid_prime(self.layers[-1].z)
        outputs = [np.array(x)] # this array takes the layers after activation function
        zs = []  # this array takes the layers before activation function
        for l in self.layers:
            zs.append(l.z)
            outputs.append(l.output)
        b_change[-1] = delta
        w_change[-1] = np.dot(delta, outputs[-2].T)
        for l in range(2, len(self.layers)+1):
            prime = sigmoid_prime(zs[-l])
            delta = np.dot(self.layers[-l+1].weights.T, delta) * prime
            b_change[-l] = delta
            w_change[-l] = np.dot(delta, outputs[-l-1].T)
        return w_change, b_change

    # this is a simplified version of the train method, for now it works with just 1 input
    def train(self, x, y, it, alpha):
        accL1 = []
        for i in range(it):
            dw, db = t.backprop(x, y)
            for (l, w, b) in zip(self.layers, dw, db):
                l.weights = l.weights - alpha * w
                l.biases = l.biases - alpha * b
            accL1.append(L1(self.layers[-1].output, y))
        return self.feedforward(x), accL1



""" 
    Testing section
"""

iterations = 100
alpha = 0.01
t = NeuralNetworkModel(3, 2)
t.add(Dense(3, 4, "sigmoid"))
t.add(Dense(4, 2, "softmax"))
result, x = t.train([[2], [3], [2.5]], [[0], [1]], iterations, alpha)
plt.suptitle("Neural Net")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.plot(x)
plt.show()
print(result)
a = t.feedforward([[6],[1],[10]])
print(a)
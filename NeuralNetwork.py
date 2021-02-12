import random
from Layer import Dense
from LossFunctions import *
from ActivationFunctions import *
import datetime

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
            x = self.layers[i].activation(np.dot(self.weights[i - 1], x) + self.biases[i - 1])
        return x

    def train(self, train_set, epochs, alpha, batch_size, test_set):
        for i in range(epochs):
            random.shuffle(train_set)
            batches = [train_set[k:k + batch_size] for k in range(0, len(train_set), batch_size)]
            for batch in batches:
                self.update_batch(batch, alpha)
            result = 0
            for x, y in test_set:
                output = self.feedforward(x)
                result += int(np.argmax(output) == y)
            print("Epoch {0}: {1} / {2}".format(i + 1, result, len(test_set)))

    def update_batch(self, batch, alpha):
        b_change = [np.zeros(b.shape) for b in self.biases]
        w_change = [np.zeros(w.shape) for w in self.weights]
        for (a, y) in batch:
            x = np.transpose([a])
            dw, db = self.backprop(x, y)
            b_change = [bc + dbc for bc, dbc in zip(b_change, db)]
            w_change = [wc + dwc for wc, dwc in zip(w_change, dw)]
        self.weights = [w - (alpha / len(batch)) * ndw for (w, ndw) in zip(self.weights, w_change)]
        self.biases = [b - (alpha / len(batch)) * ndb for (b, ndb) in zip(self.biases, b_change)]

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
            delta = np.dot(self.weights[-l + 1].T, delta) * prime
            b_change[-l] = delta
            w_change[-l] = np.dot(delta, outputs[-l - 1].T)
        return w_change, b_change

    def save(self, name=None):
        if name:
            f = open(name, "w+")
        else:
            day = datetime.datetime.now()
            day = str(day).split(".")[0]
            day = day.replace("-", "").replace(":", "").replace(" ", "")
            f = open("model{}.txt".format(day), "w+")

        f.write("Neural Network from Scratch\n")
        f.write("Layers: {}\n".format(len(self.sizes) - 1))
        for i in range(1, len(self.sizes)):
            layer = self.layers[i]
            f.write("Layer nr. {}: Size: {} Function: {}".format(i, self.sizes[i], layer.function))
            if layer.bounds != (-1, 1):
                f.write(" Weight Bounds: {}".format(layer.bounds))
            f.write("\n")
            f.write("Weights:\n")
            for x in self.weights[i-1]:
                for y in x:
                    f.write("{} ".format(y))
                f.write("\n")
            f.write("\n")
            f.write("Biases:\n")
            for x in self.biases[i-1]:
                for y in x:
                    f.write("{} ".format(y))
                f.write("\n")
            f.write("\n")
        f.close()


"""
    For testing purposes
"""
a = NeuralNetworkModel(10, 1)
a.add(Dense(5, "sigmoid", weightBounds=(0, 1)))
a.add(Dense(1, "sigmoid"))
a.save("model.txt")
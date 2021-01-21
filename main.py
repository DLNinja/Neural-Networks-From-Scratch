"""
    This is the file where I'll use everything I built to create a NN to do something (idk atm)
"""

from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
from Layer import *
from keras.datasets import mnist  # for later

# Should be working, but it doesn't

iterations = 10
alpha = 0.1
t = NeuralNetworkModel(784, 10)
t.add(Dense(64, "sigmoid"))
t.add(Dense(10, "relu"))
t.add(Dense(10, "sigmoid"))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 784))/255
x_test = x_test.reshape((-1, 784))/255

def categorize(a):
    e = np.zeros((10, 1))
    e[a] = 1
    return e

# I'll train it on a small part of the dataset 'cause my laptop can't do more
# but now I run the entire dataset on colab
x = x_train[:500]
y = []
for j in y_train[:500]:
    y.append(categorize(j))

t.train(x_train, y, iterations, alpha, 32, x_test, y_test)

""" The Results after running it in colab:


With only 2 layers - it ran better
t.add(Dense(64, "relu"))
t.add(Dense(10, "sigmoid"))

Epoch 1: 2588 / 10000
Epoch 2: 3056 / 10000
Epoch 3: 3530 / 10000
Epoch 4: 3880 / 10000
Epoch 5: 4169 / 10000
Epoch 6: 4359 / 10000
Epoch 7: 4556 / 10000
Epoch 8: 4716 / 10000
Epoch 9: 4866 / 10000
Epoch 10: 4992 / 10000

Almost 50 %

And after 10 more epochs:

Epoch 11: 5103 / 10000
Epoch 12: 5231 / 10000
Epoch 13: 5334 / 10000
Epoch 14: 5446 / 10000
Epoch 15: 5545 / 10000
Epoch 16: 5649 / 10000
Epoch 17: 5772 / 10000
Epoch 18: 5884 / 10000
Epoch 19: 5990 / 10000
Epoch 20: 6103 / 10000

"""

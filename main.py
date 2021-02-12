"""
    This is the file where I'll use everything I built to create a NN to do something (idk atm)
"""

from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
from Layer import *
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 784))/255
x_test = x_test.reshape((-1, 784))/255

def categorize(j):
    a = np.zeros((10, 1))
    a[j] = 1
    return a


y = []
for j in y_train:
    y.append(categorize(j))

train = list(zip(x_train, y))
test = list(zip(x_test, y_test))

epochs = 20
alpha = 0.1
batch_size = 32
t = NeuralNetworkModel(784, 10)
t.add(Dense(64, "sigmoid"))
t.add(Dense(10, "sigmoid"))

t.train(train, epochs, alpha, batch_size, test)

""" The Results after running it:

On my laptop, with the above structure:

Epoch 1: 1716 / 10000
Epoch 2: 2849 / 10000
Epoch 3: 3964 / 10000
Epoch 4: 4184 / 10000
Epoch 5: 4299 / 10000
Epoch 6: 4606 / 10000
Epoch 7: 4968 / 10000
Epoch 8: 5094 / 10000
Epoch 9: 5160 / 10000
Epoch 10: 5205 / 10000
Epoch 11: 5243 / 10000
Epoch 12: 5276 / 10000
Epoch 13: 5303 / 10000
Epoch 14: 5317 / 10000
Epoch 15: 5330 / 10000
Epoch 16: 5339 / 10000
Epoch 17: 5350 / 10000
Epoch 18: 5358 / 10000
Epoch 19: 5366 / 10000
Epoch 20: 5373 / 10000

Process finished with exit code 0


In colab, with relu and sigmoid, 20 epochs

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

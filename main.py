"""
    This is the file where I'll use everything I built to create a NN to do something (idk atm)
"""

from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
from Layer import *
from keras.datasets import mnist # for later

# Should be working, but it doesn't

iterations = 100
alpha = 0.1
t = NeuralNetworkModel(784, 10)
t.add(Dense(64, "relu"))
t.add(Dense(10, "relu"))
t.add(Dense(10, "softmax"))
#x = t.train(x_train, y_train, iterations, alpha)

# plt.suptitle("Neural Net")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.plot(x)
# plt.show()


# plt.figure(figsize=(9, 3))
# plt.subplot(121)
# plt.plot(x[0])
# plt.subplot(122)
# plt.plot(x[1])
# plt.suptitle('Losses')
# plt.show()

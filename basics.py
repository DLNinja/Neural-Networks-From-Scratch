"""
  This is a little introduction to Neural Networks, how they work and a little bit of basic code

 A Neural Network is a learning system which resembles the human brain, built with lots of elements called neurons or perceptrons.
 Neurons are arranged in layers: input layer, hidden layers and output layer.
 The input layer takes the input of the model and feeds it to the next layer through connections named weights,
 and so on, until it reaches the output layer, where it will generate predictions

 The input layer is a vector which contains values like senzor data, coordinates or pixels of an image (and many more examples)
"""

# This an example of a simple NN, with just the input and output layers, without any hidden layers
# The input layer contains 3 variables and the NN will output one variable

inputs = [2, 1.4, 0.9]
weights = [0.4, 0.2, 3]

# The bias is just a neuron with a fixed variable which doesnt have any connections with the prior layer, only with the next one
bias = 1

# The neuron from the output layer will take the sum of each neuron from the prior layer multiplied by their weight

output = 0
for neuron, weight in zip(inputs, weights):
    output += neuron * weight
output += bias
# print(output) # prints 4.78

# Or we can do it easier using numpy
import numpy as np

output = np.dot(inputs, weights) + bias  # this uses the dot product, a mathematical operation, used in vector and matrix multiplication
print(output)  # prints 4.78, like before, but in 2 lines

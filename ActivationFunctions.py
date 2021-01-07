"""
    This is the file that contains the Activation Functions

    Functions implemented:
        - ReLU
        - Sigmoid
        - TanH
    To be implemented:
        - Softmax

"""

import numpy as np

"""
    The ReLU function outputs the maximum between a value and 0
        f(x) = max(x, 0)
    The ReLU function is a simple function, but in can do a lot of things
"""

def ReLU(layer):
    return np.array([max(i, 0) for i in layer])


"""
    Sigmoid function looks something like this:
        f(x) = 1/(1+exp(-x))
    This function will output values between 0 and 1
"""

def sigmoid(layer):
    exp = np.array([np.e**(-x) for x in layer])
    return np.array([1/(1+x) for x in exp])

"""
    TanH function looks something like this:
         tanh(x)= (exp(x)-exp(−x))/(exp(x)+exp(−x))
    This function will output values between -1 and 1
"""

def tanh(layer):
    result = []
    expMinus = np.array([np.e**(-x) for x in layer])
    expPlus = np.array([np.e**x for x in layer])
    for (x, y) in zip(expPlus, expMinus):
        result.append((x-y)/(x+y))
    return np.array(result)




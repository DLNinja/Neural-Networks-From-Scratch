"""
    This is the file that contains the Activation Functions

    Functions implemented:
        - ReLU
        - TanH
        - Sigmoid
        - Softmax
        (There will be more)

    You'll see me use the function exp(x), this means Euler's number, e, to the power of x: e**x, where e = 2.718...
"""

import numpy as np

"""
    The ReLU function outputs the maximum between a value and 0
        f(x) = max(x, 0)
    The ReLU function is a simple function, but in can do a lot of things if used more than once
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

"""
    The Softmax function rescales an input Tensor such that the elements are between 0 and 1 and sum to 1
    This function is often used in the output layer.
    Mathematically, it's something like this:
    X = [0, 1, 2]
    softmax(0) = exp(0)/(exp(0) + exp(1) + exp(2)), and so on
"""

def softmax(layer):
    exp = np.array([np.e**x for x in layer])
    return np.array([(np.e**x)/sum(exp) for x in layer])


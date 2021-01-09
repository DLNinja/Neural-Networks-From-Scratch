"""

    This is the file that contains all the loss functions(to be added)

"""

"""
    L1 Loss function stands for Least Absolute Deviations. Also known as LAD
    L1 is used to minimize the error
    It is the sum of the all the absolute differences between the true value and the predicted value
"""

def L1(predictedLayer, trueLayer):
    return sum(abs(true-predicted) for (true, predicted) in zip(trueLayer, predictedLayer))


"""
    L2 Loss function stands for Least Square Errors. Also known as LS
    L2 is like L1, but the differences are squared, so there's no need for the absolute function.
"""

def L2(predictedLayer, trueLayer):
    return sum((true-predicted)**2 for (true, predicted) in zip(trueLayer, predictedLayer))


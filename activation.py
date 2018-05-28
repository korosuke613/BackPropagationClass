import math


class Sigmoid:
    # sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    @staticmethod
    def dsigmoid(y):
        return y * (1.0 - y)


class Tanh:
    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    @staticmethod
    def tanh(x):
        return math.tanh(x)

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    @staticmethod
    def dtanh(y):
        return 1.0 - y ** 2

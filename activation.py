import math
from abc import ABC, abstractmethod


class Activator(ABC):
    @staticmethod
    @abstractmethod
    def activate(x):
        pass

    @staticmethod
    @abstractmethod
    def differential(x):
        pass


class Sigmoid(Activator):
    # sigmoid function
    @staticmethod
    def activate(x):
        return 1.0 / (1.0 + math.exp(-x))

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    @staticmethod
    def differential(x):
        return x * (1.0 - x)


class Tanh(Activator):
    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    @staticmethod
    def activate(x):
        return math.tanh(x)

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    @staticmethod
    def differential(x):
        return 1.0 - x ** 2

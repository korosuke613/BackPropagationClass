import random


# calculate a random number where:  a <= rand < b
def uniform(a, b):
    return (b - a) * random.random() + a


def normalvariate(start=0.0, end=1.0, coefficient=0.1):
    return coefficient * random.normalvariate(start, end)


def gauss(start=0.0, end=0.2, coefficient=0.1):
    return coefficient * random.gauss(start, end)

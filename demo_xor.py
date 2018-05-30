from neural_network import NeuralNetwork
from activation import Sigmoid


def demo():
    # Teach network XOR function
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NeuralNetwork(2, 2, 1, activator=Sigmoid, title='XOR')
    # train it with some patterns
    n.train(pat, epoch=1000)
    n.print_error(is_graph=True)
    # test it
    n.test(pat)


if __name__ == '__main__':
    demo()

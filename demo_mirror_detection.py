from neural_network import NeuralNetwork
from activation import Sigmoid
import matplotlib.pyplot as plt
from tqdm import tqdm


class NnMirror(NeuralNetwork):
    def __init__(self, ni, nh, no, pattern, activator=Sigmoid, title='none'):
        super().__init__(ni, nh, no, activator, title)
        self.pattern = pattern

    def draw(self):
        def draw_surface():
            result = []
            for p in tqdm(self.pattern):
                result.append(self.update(p[0]))
            draw_p = [(p, result[i][0]) for i, p in enumerate(self.pattern)]
            draw_x = [i for i, x in enumerate(draw_p)]
            draw_ans = [a[1] for a in draw_p]
            plt.scatter(draw_x, draw_ans, c=draw_ans, cmap=plt.cm.Spectral, s=15)
            plt.plot(draw_x, draw_ans)

        plt.title(self.get_plt_title())
        draw_surface()

        plt.xticks([0, 16, 32, 48, 63], ["000000", "001000", "010000", "110000", "111111"])
        plt.savefig(self.timestamp + '.draw.pdf')
        plt.show()


def demo():
    # Teach network symmetry detecting function
    pat = [
        [[0, 0, 0, 0, 0, 0], [1]],
        [[0, 0, 0, 0, 0, 1], [0]],
        [[0, 0, 0, 0, 1, 0], [0]],
        [[0, 0, 0, 0, 1, 1], [0]],
        [[0, 0, 0, 1, 0, 0], [0]],
        [[0, 0, 0, 1, 0, 1], [0]],
        [[0, 0, 0, 1, 1, 0], [0]],
        [[0, 0, 0, 1, 1, 1], [0]],
        [[0, 0, 1, 0, 0, 0], [0]],
        [[0, 0, 1, 0, 0, 1], [0]],
        [[0, 0, 1, 0, 1, 0], [0]],
        [[0, 0, 1, 0, 1, 1], [0]],
        [[0, 0, 1, 1, 0, 0], [1]],
        [[0, 0, 1, 1, 0, 1], [0]],
        [[0, 0, 1, 1, 1, 0], [0]],
        [[0, 0, 1, 1, 1, 1], [0]],
        [[0, 1, 0, 0, 0, 0], [0]],
        [[0, 1, 0, 0, 0, 1], [0]],
        [[0, 1, 0, 0, 1, 0], [1]],
        [[0, 1, 0, 0, 1, 1], [0]],
        [[0, 1, 0, 1, 0, 0], [0]],
        [[0, 1, 0, 1, 0, 1], [0]],
        [[0, 1, 0, 1, 1, 0], [0]],
        [[0, 1, 0, 1, 1, 1], [0]],
        [[0, 1, 1, 0, 0, 0], [0]],
        [[0, 1, 1, 0, 0, 1], [0]],
        [[0, 1, 1, 0, 1, 0], [0]],
        [[0, 1, 1, 0, 1, 1], [0]],
        [[0, 1, 1, 1, 0, 0], [0]],
        [[0, 1, 1, 1, 0, 1], [0]],
        [[0, 1, 1, 1, 1, 0], [1]],
        [[0, 1, 1, 1, 1, 1], [0]],
        [[1, 0, 0, 0, 0, 0], [0]],
        [[1, 0, 0, 0, 0, 1], [1]],
        [[1, 0, 0, 0, 1, 0], [0]],
        [[1, 0, 0, 0, 1, 1], [0]],
        [[1, 0, 0, 1, 0, 0], [0]],
        [[1, 0, 0, 1, 0, 1], [0]],
        [[1, 0, 0, 1, 1, 0], [0]],
        [[1, 0, 0, 1, 1, 1], [0]],
        [[1, 0, 1, 0, 0, 0], [0]],
        [[1, 0, 1, 0, 0, 1], [0]],
        [[1, 0, 1, 0, 1, 0], [0]],
        [[1, 0, 1, 0, 1, 1], [0]],
        [[1, 0, 1, 1, 0, 0], [0]],
        [[1, 0, 1, 1, 0, 1], [1]],
        [[1, 0, 1, 1, 1, 0], [0]],
        [[1, 0, 1, 1, 1, 1], [0]],
        [[1, 1, 0, 0, 0, 0], [0]],
        [[1, 1, 0, 0, 0, 1], [0]],
        [[1, 1, 0, 0, 1, 0], [0]],
        [[1, 1, 0, 0, 1, 1], [1]],
        [[1, 1, 0, 1, 0, 0], [0]],
        [[1, 1, 0, 1, 0, 1], [0]],
        [[1, 1, 0, 1, 1, 0], [0]],
        [[1, 1, 0, 1, 1, 1], [0]],
        [[1, 1, 1, 0, 0, 0], [0]],
        [[1, 1, 1, 0, 0, 1], [0]],
        [[1, 1, 1, 0, 1, 0], [0]],
        [[1, 1, 1, 0, 1, 1], [0]],
        [[1, 1, 1, 1, 0, 0], [0]],
        [[1, 1, 1, 1, 0, 1], [0]],
        [[1, 1, 1, 1, 1, 0], [0]],
        [[1, 1, 1, 1, 1, 1], [1]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NnMirror(6, 2, 1, pat, activator=Sigmoid, title='Mirror')
    # train it with some patterns
    # n.activation = tanh
    # n.dactivation = dtanh
    n.train(pat, epoch=60)
    n.test(pat)
    n.print_error(is_graph=True)
    n.draw()
    # test it


if __name__ == '__main__':
    demo()

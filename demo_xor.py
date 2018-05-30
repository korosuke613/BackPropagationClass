from neural_network import NeuralNetwork
from activation import Sigmoid
import matplotlib.pyplot as plt
from tqdm import tqdm


class NnXor(NeuralNetwork):
    def __init__(self, ni, nh, no, pattern, activator=Sigmoid, title='none'):
        super().__init__(ni, nh, no, activator, title)
        self.pattern = pattern

    def draw(self):
        def draw_pattern():
            for i in [(0, 'red'), (1, 'blue')]:
                x_set = [x[0][0] for x in self.pattern if x[1][0] == i[0]]
                y_set = [y[0][1] for y in self.pattern if y[1][0] == i[0]]
                plt.scatter(x_set, y_set, c=i[1])

        def draw_surface():
            def generate_all_patterns():
                x_set = [x for x in range(-50, 150)]
                y_set = [y for y in range(-50, 150)]
                learn_data = [[x / 100, y / 100] for x in x_set for y in y_set]
                return learn_data
            patterns = generate_all_patterns()
            result = []
            for p in tqdm(patterns):
                result.append(self.update(p))
            draw_p = [(p, result[i][0]) for i, p in enumerate(patterns)]
            draw_x = [x[0][0] for x in draw_p]
            draw_y = [y[0][1] for y in draw_p]
            draw_ans = [a[1] for a in draw_p]
            plt.scatter(draw_x, draw_y, c=draw_ans, cmap=plt.cm.Spectral, s=15)
            plt.colorbar()

        plt.title(self.get_plt_title())
        draw_surface()
        draw_pattern()
        plt.savefig('fig/' + self.timestamp + '.draw.pdf')
        plt.show()


def demo():
    # Teach network XOR function
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NnXor(2, 2, 1, pat, activator=Sigmoid, title='XOR')
    # train it with some patterns
    n.train(pat, epoch=1000)
    n.test(pat)
    n.print_error(is_graph=True)
    n.draw()
    # test it


if __name__ == '__main__':
    demo()

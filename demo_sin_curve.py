from neural_network import NeuralNetwork
from activation import Sigmoid
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random_tool


class NnSin(NeuralNetwork):
    def __init__(self, ni, nh, no, pattern, activator=Sigmoid, title='none'):
        super().__init__(ni, nh, no, activator, title)
        self.pattern = pattern

    @staticmethod
    def generate_all_patterns():
        x_set_ = [x for x in range(-300, 300)]
        y_set_ = [y for y in range(-75, 75)]
        learn_data = [[x / 50, y / 50] for x in x_set_ for y in y_set_]
        return learn_data

    def draw(self):
        def draw_surface(pattern):
            result = []
            for p in tqdm(pattern):
                result.append(self.update(p))
            draw_p = [(p, result[i][0]) for i, p in enumerate(pattern)]
            draw_x = [x[0][0] for x in draw_p]
            draw_y = [y[0][1] for y in draw_p]
            draw_ans = [a[1] for a in draw_p]
            plt.scatter(draw_x, draw_y, c=draw_ans, cmap=plt.cm.Spectral, s=15)
            plt.colorbar()

        patterns = self.generate_all_patterns()
        draw_surface(patterns)

        for i in [(0, 'red'), (1, 'blue')]:
            x_set = [x[0][0] for x in self.pattern if x[1][0] == i[0]]
            y_set = [y[0][1] for y in self.pattern if y[1][0] == i[0]]
            plt.scatter(x_set, y_set, c=i[1])
        plt.title(self.get_plt_title())
        self.draw_sin()
        plt.savefig('fig/' + self.timestamp + '.draw.pdf')
        plt.show()

    @staticmethod
    def draw_sin():
        x_set = [x / 10.0 for x in range(-60, 60)]
        y_set = [math.sin(math.pi / 2 * x) for x in x_set]
        plt.plot(x_set, y_set)


class NnSinChallenge(NnSin):
    def __init__(self, ni, nh, no, pattern, activator=Sigmoid, title='none'):
        super().__init__(ni, nh, no, pattern, activator, title)
        self.pattern = pattern

    @staticmethod
    def generate_all_patterns():
        x_set_ = [x for x in range(-450, 450)]
        y_set_ = [y for y in range(-75, 75)]
        learn_data = [[x / 50, y / 50] for x in x_set_ for y in y_set_]
        return learn_data

    @staticmethod
    def draw_sin():
        x_set = [x / 10.0 for x in range(-90, 90)]
        y_set = [math.sin(math.pi / 2 * x) for x in x_set]
        plt.plot(x_set, y_set)


def demo_sin_curve():
    def generate_leaning_data():
        learn_data = []
        up_num = down_num = 0
        while True:
            x = random_tool.uniform(-6.0, 6.0)
            y = random_tool.uniform(-1.5, 1.5)
            sin_y = math.sin(math.pi / 2 * x)
            up_correct = 0
            if y >= sin_y:
                up_correct = 1

            if up_correct == 1 and up_num < 50:
                up_num += 1
                learn_data.append([[x, y], [up_correct]])
            elif up_correct == 0 and down_num < 50:
                down_num += 1
                learn_data.append([[x, y], [up_correct]])

            if up_num >= 50 and down_num >= 50:
                break
        return learn_data

    pat = generate_leaning_data()

    n = NnSin(2, 16, 1, pat, activator=Sigmoid, title='sin_curve')
    # n.activation = tanh
    # n.dactivation = dtanh
    n.train(pat, epoch=100)
    n.test(pat)
    n.print_error(is_graph=True)
    n.draw()


if __name__ == '__main__':
    demo_sin_curve()

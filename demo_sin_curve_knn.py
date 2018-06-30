from demo_sin_curve import NnSin
import math
import random_tool
import operator
import matplotlib.pyplot as plt
from datetime import datetime


class NnSinKnn(NnSin):
    def __init__(self, pattern, k=3, title='none', fineness=40):
        super().__init__(2, 16, 1, pattern=pattern, title=title, fineness=fineness)
        self.pattern = pattern
        self.k = k
        self.bias = None
        self.variance = None
        self.error = None

    def update(self, inputs):
        euclid = []
        for i, p in enumerate(self.pattern):
            e = math.sqrt((p[0][0] - inputs[0]) ** 2 + (p[0][1] - inputs[1]) ** 2)
            euclid.append((i, e))
        euclid.sort(key=operator.itemgetter(1))
        correct = [0, 0]
        for i in range(self.k):
            c = self.pattern[euclid[i][0]][1]
            correct[c[0]] += 1
        if correct[0] > correct[1]:
            return [0]
        elif correct[0] < correct[1]:
            return [1]

        return [0.5]

    def test(self, patterns):
        def calc_bias(di):
            r = 0
            for d in di:
                r += d
            return (r / 100) ** 2

        def calc_varience(di):
            r = 0
            fx = 0
            for d in di:
                fx += d
            fx /= 100
            for d in di:
                r += (d - fx) ** 2
            return r / 100

        def calc_error(di, pat):
            r = 0
            for i, d in enumerate(di):
                r += (d - pat[i][1][0]) ** 2
            return r / 100
        correct = 0
        diffs = []
        fx = []
        for p in patterns:
            result = self.update(p[0])
            #print(p[0], '->', result)
            diff = abs(result[0] - p[1][0])
            diffs.append(p[0][1] - result[0])
            fx.append(result[0])
            if diff < 0.05:
                correct += 1
        self.bias = round(calc_bias(diffs), 5)
        self.variance = round(calc_varience(fx), 5)
        self.error = round(calc_error(fx, patterns), 5)
        self.accuracy = correct / len(patterns)
        print(f"accuracy: {self.accuracy:.2%}\n"
              f'error: {self.error}\n'
              f'bias: {self.bias}\n'
              f'variance: {self.variance}\n')

    def get_plt_title(self):
        activation = self.activator.__name__
        return f'title:{self.title}, \n' \
               f'perceptron=[{self.ni-1}, {self.nh-1}, {self.no}], k={self.k}\n' \
               f'activation={activation}, accuracy={self.accuracy*100}%\n' \
               f'error={self.error}, bias={self.bias}, variance={self.variance}'


def demo_sin_curve_knn(num, isDraw=True):
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

    n = NnSinKnn(pat, k=num, title='sin_curve_with_kNN', fineness=25)
    # n.activation = tanh
    # n.dactivation = dtanh
    n.test(pat)
    if isDraw:
        n.draw()
    return n.bias, n.variance, n.error


def create_bve():
    bias = []
    variance = []
    error = []
    for i in range(1, 11):
        print(f"k={i}")
        b = v = e = 0
        for j in range(100):
            b_, v_, e_ = demo_sin_curve_knn(i, isDraw=False)
            b += b_
            v += v_
            e += e_
        bias.append(b / 100)
        variance.append(v / 100)
        error.append(e / 100)
    plt.plot(bias)
    plt.plot(variance)
    plt.plot(error)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('fig/' + timestamp + '_' + "bias_variance" + '.draw.png')
    plt.show()


if __name__ == '__main__':
    create_bve()
    #demo_sin_curve_knn(4)

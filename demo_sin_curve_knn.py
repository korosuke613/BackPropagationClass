from demo_sin_curve import NnSin
import math
import random_tool
import operator


class NnSinKnn(NnSin):
    def __init__(self, pattern, k=3, title='none'):
        super().__init__(2, 16, 1, pattern=pattern, title=title)
        self.pattern = pattern
        self.k = k

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


def demo_sin_curve_knn():
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

    n = NnSinKnn(pat, k=3, title='sin_curve_with_kNN')
    # n.activation = tanh
    # n.dactivation = dtanh
    n.draw()


if __name__ == '__main__':
    demo_sin_curve_knn()

from activation import Sigmoid, Tanh
import random_tool
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime


class NeuralNetwork(Sigmoid, Tanh):
    def __init__(self, ni, nh, no, title='none'):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh + 1  # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = self.make_matrix(self.ni, self.nh, is_random=True)
        self.wo = self.make_matrix(self.nh, self.no, is_random=True)

        # create last change in weights for momentum
        self.ci = self.make_matrix(self.ni, self.nh)
        self.co = self.make_matrix(self.nh, self.no)

        # set activation
        self.activation = self.sigmoid
        self.dactivation = self.dsigmoid

        self.errors = []
        self.title = title
        self.iteration_num = None
        self.epoch = None
        plt.interactive(False)

    # Make a matrix (we could use NumPy to speed this up)
    @staticmethod
    def make_matrix(col, row, fill=0.0, is_random=False):
        m = []
        for i in range(col):
            if is_random is True:
                fill_matrix = []
                for j in range(row):
                    fill_matrix.append(random_tool.normalvariate())
            else:
                fill_matrix = [fill] * row
            m.append(fill_matrix)
        return m

    def update(self, inputs, activation=None):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        if activation is None:
            activation = self.activation

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            total = 0.0
            for i in range(self.ni):
                total = total + self.ai[i] * self.wi[i][j]
            self.ah[j] = activation(total)

        # output activations
        for k in range(self.no):
            total = 0.0
            for j in range(self.nh):
                total = total + self.ah[j] * self.wo[j][k]
            self.ao[k] = activation(total)

        return self.ao[:]

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))
        self.print_weights()

    def print_weights(self):
        print()
        print('Input weights:')
        for i in range(self.ni):
            print([self.wi[i][index] for index in range(len(self.wi[i]) - 1)])
        print()
        print('Output weights:')
        for i in range(self.nh):
            print(self.wo[i])

    def train(self, patterns, epoch=1000,  mu=0.1, velocity=0.9):
        # N: learning rate
        # M: momentum factor
        self.epoch = epoch
        self.iteration_num = self.epoch * len(patterns)
        for i in tqdm(range(self.iteration_num)):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, mu, velocity)
            if i % len(patterns) == 0:
                self.errors.append(error)

    def back_propagate(self, targets, mu, velocity, dactivation=None):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        if dactivation is None:
            dactivation = self.dactivation

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dactivation(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh - 1):
            # for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dactivation(self.ah[j]) * error

        # update output weights
        self.update_weight(self.nh, self.no, output_deltas, mu, velocity, self.ah, self.wo, self.co)

        # update input weights
        self.update_weight(self.ni, self.nh, hidden_deltas, mu, velocity, self.ai, self.wi, self.ci)

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    @staticmethod
    def update_weight(first, second, deltas, mu, velocity, a_list, w_list, c_list):
        # update output weights
        for i in range(first):
            for j in range(second):
                change = deltas[j] * a_list[i]
                w_list[i][j] = w_list[i][j] + mu * change + velocity * c_list[i][j]
                c_list[i][j] = mu * change + velocity * c_list[i][j]

    def print_error(self, is_graph=False):
        if is_graph is True:
            x = [x for x, _ in enumerate(self.errors)]
            plt.plot(x, self.errors)
            plt.xlabel("epoch")
            plt.ylabel("error")
            plt.title(self.get_plt_title())
            plt.ylim(0.0, max(self.errors) + 1.0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig('fig/' + timestamp + '_' + self.title + '.pdf')
            plt.show()
        else:
            for error in self.errors:
                print('error %-.5f' % error)

    def get_plt_title(self):
        activation, dactivation = self.get_activation()
        return f'title:{self.title}, \n' \
               f'perceptron=[{self.ni-1}, {self.nh-1}, {self.no}], ' \
               f'epoch={self.epoch} iter={self.iteration_num}, \n'\
               f'activation={activation}'

    def get_activation(self):
        if self.activation is self.tanh:
            return "tanh", 'dtanh'
        return "sigmoid", 'dsigmoid'
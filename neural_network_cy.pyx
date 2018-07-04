from activation import Activator, Sigmoid
import random_tool
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from typing import Type
import time
from libcpp cimport bool


cdef class NeuralNetwork:
    cdef list ai, ah, ao, wi, wo, ci, co, errors
    cdef int ni, nh, no, iteration_num, epoch, time_elapsed
    cdef double accuracy
    cdef object activator
    cdef str plt_title

    def __init__(self, int ni, int nh, int no, object activator: Type[Activator]=Sigmoid, str title='none'):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh + 1  # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = self.make_matrix(self.ni, self.nh, 0.0, is_random=True)
        self.wo = self.make_matrix(self.nh, self.no, 0.0, is_random=True)

        # create last change in weights for momentum
        self.ci = self.make_matrix(self.ni, self.nh, 0.0, is_random=True)
        self.co = self.make_matrix(self.nh, self.no, 0.0, is_random=True)

        # set activation
        self.activator = activator
        self.accuracy = 0.0
        self.errors = []
        self.title = title
        self.plt_title = ""
        self.iteration_num = -1
        self.epoch = -1
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.time_elapsed = 0
        plt.interactive(False)
        plt.subplots_adjust(top=0.8)

    # Make a matrix (we could use NumPy to speed this up)
    cdef list make_matrix(self, int col, int row, double fill, bool is_random):
        cdef list fill_matrix, m = []
        cdef int i, j
        for i in range(col):
            if is_random is True:
                fill_matrix = []
                for j in range(row):
                    fill_matrix.append(random_tool.normalvariate())
            else:
                fill_matrix = [fill] * row
            m.append(fill_matrix)
        return m

    cpdef list update(self, list inputs):
        cdef int i, j, k
        cdef double total

        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            total = 0.0
            for i in range(self.ni):
                total = total + self.ai[i] * self.wi[i][j]
            self.ah[j] = self.activator.activate(total)

        # output activations
        for k in range(self.no):
            total = 0.0
            for j in range(self.nh):
                total = total + self.ah[j] * self.wo[j][k]
            self.ao[k] = self.activator.activate(total)

        return self.ao[:]

    def test(self, patterns):
        correct = 0
        for p in patterns:
            result = self.update(p[0])
            print(p[0], '->', result)
            if abs(result[0] - p[1][0]) < 0.05:
                correct += 1
        self.print_weights()
        self.accuracy = correct / len(patterns)
        print("accuracy: {:.2%}".format(self.accuracy))

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
        self.epoch = epoch
        self.iteration_num = self.epoch * len(patterns)
        time_start = time.time()
        for i in tqdm(range(self.iteration_num)):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, mu, velocity)
            if i % len(patterns) == 0:
                self.errors.append(error)
        self.time_elapsed = time.time() - time_start

    cdef double back_propagate(self, list targets, double mu, double velocity):
        cdef list output_deltas, hidden_deltas
        cdef int j, k
        cdef double error

        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = self.activator.differential(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh - 1):
            # for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = self.activator.differential(self.ah[j]) * error

        # update output weights
        self.update_weight(self.nh, self.no, output_deltas, mu, velocity, self.ah, self.wo, self.co)

        # update input weights
        self.update_weight(self.ni, self.nh, hidden_deltas, mu, velocity, self.ai, self.wi, self.ci)

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    cdef update_weight(self, int first, int second, list deltas, double mu, double velocity,
                       list a_list, list w_list, list c_list):
        # update output weights
        cdef int i, j
        cdef double change
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
            plt.savefig('fig/' + self.timestamp + '_' + self.title + '.pdf')
            plt.show()
        else:
            for error in self.errors:
                print('error %-.5f' % error)

    def get_plt_title(self):
        activation = self.activator.__name__
        return f'title:{self.title}, \n' \
               f'perceptron=[{self.ni-1}, {self.nh-1}, {self.no}], ' \
               f'epoch={self.epoch}, iter={self.iteration_num}, \n'\
               f'activation={activation},  accuracy={self.accuracy*100}%, time={round(self.time_elapsed, 2)}s'

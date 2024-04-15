import os
import sys
sys.path.append('.'+ os.sep + 'private')
import private.neural_network as nn
import private.functional as fnl
import private.my_tensor as mtr
import numpy as np
import train as tr
import mnist_reader as mr
# import matplotlib as mpl
# import matplotlib.pyplot as plt

class test_nn(nn.neural_network):
    def __init__(self):
        super().__init__()

        nn.layers(self, [
            fnl.LinFC(784, 128),
            fnl.ReLU(128, 128),
            fnl.LinFC(128, 64),
            fnl.ReLU(64, 64),
            fnl.LinFC(64, 10),
            fnl.Softmax(10, 10)
        ])

epoch = 18
tnn = test_nn()
# tnn.init()

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'

x_data, y_data = mr.load_mnist(data_path)
x_test_data, y_test_data = mr.load_mnist(data_path, kind='t10k')

x_data = x_data.reshape(*x_data.shape, 1)
x_data = x_data[:2048]/255
x_data = x_data - np.mean(x_data, axis=0)

x_test_data = x_test_data.reshape(*x_test_data.shape, 1)
x_test_data = x_test_data[:10000]/255
x_test_data = x_test_data - np.mean(x_data, axis=0)
# x_data /= np.var(x_data, axis=0)
y_data = y_data[:2048]
y_test_data = y_test_data[:10000]

# np.random.randn((2, 3))

tr.train(tnn, x_data, y_data, 10 , 64, epoch, 0.1, lr=0.6)
tr.test(tnn, x_test_data, y_test_data)
# tnn.fval(x_data[0].reshape(len(x_data[0]), 1))

# print((C@B@A@x))

# tnn.layers[-1].ahead.backward()
# print(tnn.layers[0].diff)
# print(C@B[:, 0]@[x[0, 0]])
import sys
sys.path.append('./private')
import private.neural_network as nn
import private.functional as fnl
import private.my_tensor as mtr
import numpy as np
import train as tr
import mnist_reader as mr
class test_nn(nn.neural_network):
    def __init__(self):
        super().__init__()
        self.layers = [
            fnl.LinFC(784, 128),
            fnl.ReLU(128, 128),
            fnl.LinFC(128, 32),
            fnl.ReLU(32, 32),
            fnl.LinFC(32, 10),
            fnl.Softmax(10, 10)
        ]

    def forward(self):
        for layer in self.layers:
            layer.fval()

    def get_x(self):
        return self.layers[0].back[0]

    def fval(self, x):
        self.layers[0].back[0].val = x
        self.forward()

        return self.layers[-1].ahead
    
    def zero_diff(self):
        for layer in self.layers:
            if layer.parameter is not None:
                layer.zero_diff()
    
    def ahead(self):
        return self.layers[-1].ahead
    
    def update(self, lr):
        for layer in self.layers:
            if layer.parameter is not None:
                layer.parameter += -lr * layer.diff

    def init(self):
        start = mtr.my_tensor(np.random.randn(self.layers[0].in_dim, 1))
        back = start
        for layer in self.layers:
            for i in range(layer.back_num):
                layer.back.append(back)
                if i >= 1:
                    layer.back.append(self.layers[layer.extra_back[i-1]].back[0])
            back = layer.ahead

epoch = 5
tnn = test_nn()
tnn.init()
x_data, y_data = mr.load_mnist('/home/yezq/myproject/NeuralNetwork_HW1/data/fashion/')
x_data = x_data[:800]/255
x_data = x_data - np.mean(x_data, axis=0)
# x_data /= np.var(x_data, axis=0)
y_data = y_data[:800]

# print(np.mean(x_data, axis=0))
# print(np.var(x_data, axis=0))

# print(np.log(tnn.fval(x_data[0]).val))

tr.train(tnn, x_data, y_data, 10 , 200, epoch, lr=0.01)



# print((C@B@A@x))

# tnn.layers[-1].ahead.backward()
# print(tnn.layers[0].diff)
# print(C@B[:, 0]@[x[0, 0]])
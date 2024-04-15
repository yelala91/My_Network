# neural_network

import functional as fnl
import numpy as np
import my_tensor as mtr

def layer_init(layers):
    start = mtr.my_tensor()
    back = start
    for layer in layers:
        for i in range(layer.back_num):
            layer.back.append(back)
            if i >= 1:
                layer.back.append(layers[layer.extra_back[i-1]].back[0])
        back = layer.ahead
    
    return layers

def layers(self, layer_list):
    self.layers = layer_init(layer_list)

class neural_network:
    def __init__(self):
        pass

    def forward(self, x):
        pass
    
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
    
    def parameter(self):
        param = []
        for layer in self.layers:
            if layer.parameter is not None:
                param.append(layer.parameter)
        return param
    
    def update(self, lr, sigma):
        for layer in self.layers:
            if layer.parameter is not None:
                layer.parameter += -lr * (layer.diff + sigma*layer.parameter)


class my_nn(neural_network):
    def __init__(self):
        super().__init__()
        self.layers = [
            fnl.LinFC(784, 128),
            fnl.LinFC(128, 32),
            fnl.LinFC(32, 10)
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.val(x)


def Loss(model, x, y, kinds_num):
    n = len(x)
    loss = np.array([[0]], dtype='float64')
    model.zero_diff()
    for i in range(n):
        L1 = fnl.PROJ(kinds_num, y[i])
        L2 = fnl.NLog((1, 1), (1, 1))
        L1.back.append(model.ahead())
        L2.back.append(L1.ahead)

        model.fval(x[i].reshape(len(x[0]), 1))
        L1.fval()
        L2.fval()

        loss += L2.ahead.val
        L2.ahead.backward(factor=n)
    loss /= n
    return loss
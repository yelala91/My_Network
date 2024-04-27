# neural_network.py
# 
# ===============================================================
# The definition of the class neural network.
# ===============================================================

import functional as fnl
import numpy as np
import my_tensor as mtr
import train as tr

# init the hidden layers
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

def layers(network, layer_list):
    network.layers = layer_init(layer_list)

class neural_network:
    def __init__(self):
        pass
    
    def forward(self):
        for layer in self.layers:
            layer.fval()

    def get_x(self):
        return self.layers[0].back[0]

    def fval(self, x):
        self.layers[0].back[0].val = x
        self.forward()

        return self.layers[-1].ahead.val
    
    def zero_diff(self):
        for layer in self.layers:
            if layer.parameter is not None:
                layer.zero_diff()
    
    def ahead(self):
        return self.layers[-1].ahead
    
    # get parameter
    def parameter(self):
        param = []
        for layer in self.layers:
            if layer.parameter is not None:
                param.append(layer.parameter)
        return param
    
    # update parameter
    def update(self, lr, sigma):
        for layer in self.layers:
            if layer.parameter is not None:
                layer.parameter += -lr * (layer.diff + sigma*layer.parameter)
    
    # save parameter            
    def save_param(self, path):
        np.savez(path, *(self.parameter()))
    
    # load parameter
    def load_param(self, param_path):
        param_dic = np.load(param_path)
        param_list = list(param_dic.values())
        i = 0
        for layer in self.layers:
            if layer.parameter is not None:
                layer.parameter = param_list[i]
                i += 1
    
    def load_param_from_list(self, param_list):
        i = 0
        for layer in self.layers:
            if layer.parameter is not None:
                layer.parameter = param_list[i]
                i += 1

class my_nn(neural_network):
    def __init__(self):
        super().__init__()

        layers(self, [
            fnl.LinFC(784, 128),
            fnl.ReLU(128, 128),
            fnl.LinFC(128, 32),
            fnl.ReLU(32, 32),
            fnl.LinFC(32, 10),
            fnl.Softmax(10, 10)
        ])
        
class my_larger_nn(neural_network):
    def __init__(self):
        super().__init__()

        layers(self, [
            fnl.LinFC(784, 256),
            fnl.ReLU(256, 256),
            fnl.LinFC(256, 64),
            fnl.ReLU(64, 64),
            fnl.LinFC(64, 10),
            fnl.Softmax(10, 10)
        ])

class my_one_layer(neural_network):
    def __init__(self):
        super().__init__()

        layers(self, [
            fnl.LinFC(784, 10),
            fnl.Softmax(10, 10)
        ])
def Loss(model, x, y, kinds_num, sigma):
    n = len(x)
    loss = np.array([[0]], dtype='float64')
    model.zero_diff()
    for xi, yi in zip(x, y):
        L1 = fnl.PROJ(kinds_num, yi)
        L2 = fnl.NLog((1, 1), (1, 1))
        L1.back.append(model.ahead())
        L2.back.append(L1.ahead)

        model.fval(xi)
        L1.fval()
        L2.fval()

        loss += np.squeeze(L2.ahead.val)
        L2.ahead.backward(factor=n)
    loss /= n
    # loss += 0.5*sigma*tr.norm_param(model.parameter())**2
    return loss


    
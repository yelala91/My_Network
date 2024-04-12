# neural_network

import functional as fnl
import numpy as np
import my_tensor as mtr

class neural_network:
    def __init__(self):
        pass

    def forward(self, x):
        pass

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
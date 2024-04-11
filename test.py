import sys
sys.path.append('./private')
import private.neural_network as nn
import private.functional as fnl
import private.my_tensor as mtr
import numpy as np

class test_nn(nn.neural_network):
    def __init__(self):
        super().__init__()
        self.layers = [
            fnl.LinFC(3, 3),
            fnl.LinFC(3, 2),
            fnl.LinFC(2, 1)
        ]

    def forward(self):
        for layer in self.layers:
            layer.fval()

    def get_x(self):
        return self.layers[0].back

    def fval(self, x):
        self.layers[0].back.val = x
        self.forward()

        return self.layers[-1].ahead

    def init(self):
        start = mtr.my_tensor(np.random.randn(self.layers[0].in_dim, 1))
        back = start
        for layer in self.layers:
            layer.back = back
            back = layer.ahead

tnn = test_nn()
tnn.init()
tnn.forward()

x = tnn.get_x().val
A = tnn.layers[0].parameter
B = tnn.layers[1].parameter
C = tnn.layers[2].parameter

print(tnn.layers[-1].ahead.val)

print((C@B@A@x))

tnn.layers[-1].ahead.backward()
print(tnn.layers[0].diff)
print(C@B[:, 0]@[x[0, 0]])
# neural_network

import functional as fnl

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
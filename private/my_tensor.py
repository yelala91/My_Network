# my_tensor.py

import numpy as np

class my_tensor:
    def __init__(self, val=None, operator=None, sub_tensor=None, type=0):
        self.val        = val
        self.operator   = operator
        self.type       = type

    def backward(self, factor=1, up_diff=np.array([[1]])):
        if self.operator is not None:
            self.operator.update_diff(up_diff/factor)
            up_diff = self.operator.grad(up_diff)
            for i in range(self.operator.back_num):
                self.operator.back[i].backward(factor=factor, up_diff=up_diff[i])

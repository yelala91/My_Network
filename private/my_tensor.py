# my_tensor.py

import numpy as np

class my_tensor:
    def __init__(self, val=None, operator=None, sub_tensor=None, type=0):
        self.val        = val
        self.operator   = operator
        self.sub_tensor = sub_tensor
        self.type       = type

    def backward(self, up_diff=np.array([[1]])):
        if self.operator is not None:
            self.operator.update_diff(up_diff)
            grad = self.operator.grad()
            up_diff = np.matmul(up_diff, grad.T)
            self.operator.back.backward(up_diff)

def tensor_add(t1, t2):
    t3 = my_tensor(val=t1.val+t2.val)
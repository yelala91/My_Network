# functional.py

import numpy as np
import my_tensor as mtr

class functional:
    # diff is the diff of the last loss about these parameter
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.diff       = diff
        self.parameter  = parameter

    def grad(self, x):
        pass

# Linear full connect functional
class LinFC(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(LinFC, self).__init__(in_dim, out_dim, parameter, diff)
        self.parameter = np.random.randn(out_dim, in_dim)
        self.diff      = np.zeros((out_dim, in_dim))
        self.ahead     = mtr.my_tensor(np.zeros((out_dim, 1)))
        self.ahead.operator = self
        self.back      = None

    def fval(self):
        x = self.back.val
        self.ahead.val = np.matmul(self.parameter, x)

    def grad(self):
        grad_x = self.parameter.T
        return grad_x
    
    def update_diff(self, up_diff):
        x = self.back.val
        self.diff += np.matmul(up_diff.T, x.T)


# functional.py

import numpy as np
import my_tensor as mtr

eps = 1e-5

class functional:
    # diff is the diff of the last loss about these parameter
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.diff       = diff
        self.parameter  = parameter
        self.back       = []

    def grad(self, x):
        pass

# Linear full connect functional
class LinFC(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(LinFC, self).__init__(in_dim, out_dim, parameter, diff)
        self.parameter      = np.random.randn(out_dim, in_dim+1)
        self.diff           = np.zeros((out_dim, in_dim+1))
        self.ahead          = mtr.my_tensor(np.zeros((out_dim, 1)))
        self.ahead.operator = self
        self.back_num       = 1

    def zero_diff(self):
        self.diff = np.zeros((self.out_dim, self.in_dim+1))

    def fval(self):
        x = self.back[0].val
        self.ahead.val = np.matmul(self.parameter[:, 1:], x) + self.parameter[:, [0]]

    def grad(self, up_diff):
        grad_x = np.matmul(up_diff, self.parameter[:, 1:])
        return [grad_x]
    
    def update_diff(self, up_diff):
        x = self.back[0].val
        self.diff[:, 1:] = self.diff[:, 1:] + np.matmul(up_diff.T, x.T)
        self.diff[:, [0]] = up_diff.T
        # self.diff += np.matmul(up_diff.T, x.T)
        
# Conv2d functional
class Conv2d(functional):
    def __init__(self, in_dim, shape, parameter=None, diff=None):
        super(Conv2d, self).__init__(in_dim, None, parameter, diff)
        self.parameter      = np.random.randn(shape)
        self.shape          = shape
        self.diff           = np.zeros(shape)
        self.ahead          = mtr.my_tensor()
        self.ahead.operator = self
        self.back_num       = 1

    def zero_diff(self):
        self.diff = np.zeros(self.shape)

    def fval(self):
        x = self.back[0].val
        self.ahead.val = np.matmul(self.parameter, x)

    def grad(self, up_diff):
        grad_x = np.matmul(up_diff, self.parameter)
        return [grad_x]
    
    def update_diff(self, up_diff):
        x = self.back[0].val
        self.diff += np.matmul(up_diff.T, x.T)

# Softmax function
class Softmax(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(Softmax, self).__init__(in_dim, out_dim, parameter, diff)
        self.ahead          = mtr.my_tensor(np.zeros((out_dim, 1)))
        self.ahead.operator = self
        self.back_num       = 1

    def fval(self):
        x               = self.back[0].val
        # x += eps
        ex              = np.exp(x-np.max(x))
        self.ahead.val  = ex/np.sum(ex)

    def grad(self, up_diff):
        x           = self.back[0].val
        ex          = np.exp(x-np.max(x))
        sum_ex      = np.sum(ex)
        ex /= sum_ex
        # sum_ex_2    = sum_ex ** 2
        this_grad_x = np.matmul(ex, ex.T)
        diag        = np.diag(ex.reshape(len(ex))) # np.diag([np.squeeze(ex_i)*sum_ex for ex_i in ex])
        this_grad_x = diag - this_grad_x
        # this_grad_x = this_grad_x/sum_ex_2
        grad_x      = np.matmul(up_diff, this_grad_x)

        return [grad_x]
    
    # def grad(self, up_diff):
    #     x           = self.back[0].val
    #     ex          = np.exp(x)
    #     sum_ex      = np.sum(ex)
    #     sum_ex_2    = sum_ex ** 2
    #     this_grad_x = np.matmul(ex, ex.T)
    #     diag        = np.diag([np.squeeze(ex_i)*sum_ex for ex_i in ex])
    #     this_grad_x = diag - this_grad_x
    #     this_grad_x = this_grad_x/sum_ex_2
    #     grad_x      = np.matmul(up_diff, this_grad_x)

    #     return [grad_x]

    def update_diff(self, up_diff):
        pass

class ReLU(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(ReLU, self).__init__(in_dim, out_dim, parameter, diff)
        self.ahead          = mtr.my_tensor(np.zeros((out_dim, 1)))
        self.ahead.operator = self
        self.back_num       = 1

    def fval(self):
        x               = self.back[0].val
        self.ahead.val  = x * (x>=0)

    def grad(self, up_diff):
        x      = self.back[0].val
        grad_x = up_diff * (x.T>=0)
        return [grad_x]

    def update_diff(self, up_diff):
        pass


class Sigmoid(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(Sigmoid, self).__init__(in_dim, out_dim, parameter, diff)
        self.ahead          = mtr.my_tensor(np.zeros((out_dim, 1)))
        self.ahead.operator = self
        self.back_num       = 1

    def fval(self):
        x               = self.back[0].val
        self.ahead.val  = 1/(1+np.exp(-x))

    def grad(self, up_diff):
        x      = self.back[0].val
        grad_x = up_diff * (np.exp(-x)/(np.power(1+np.exp(-x), 2))).T
        return [grad_x]

    def update_diff(self, up_diff):
        pass

class PROJ(functional):
    def __init__(self, in_dim, proj_index, parameter=None, diff=None):
        super(PROJ, self).__init__(in_dim, (1, 1), parameter, diff)
        self.ahead          = mtr.my_tensor(np.zeros((1, 1)))
        self.ahead.operator = self
        self.back_num       = 1
        self.proj_index     = proj_index

    def fval(self):
        x               = self.back[0].val
        self.ahead.val  = np.array([[x[self.proj_index, 0]]])

    def grad(self, up_diff):
        grad_x = np.zeros((1, self.in_dim))
        grad_x[0, self.proj_index] = 1
        grad_x = np.squeeze(up_diff) * grad_x
        return [grad_x]

    def update_diff(self, up_diff):
        pass

class NLog(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(NLog, self).__init__(in_dim, out_dim, parameter, diff)
        self.ahead          = mtr.my_tensor(np.zeros(out_dim))
        self.ahead.operator = self
        self.back_num       = 1

    def fval(self):
        x               = self.back[0].val
        x += eps
        self.ahead.val  = -np.log(x)

    def grad(self, up_diff):
        x      = self.back[0].val
        x += eps
        grad_x = -1/x
        grad_x = up_diff * grad_x
        return [grad_x]

    def update_diff(self, up_diff):
        pass

# class Loss(functional):
#     def __init__(self, in_dim, N, labels, parameter=None, diff=None):
#         super(TensorAdd, self).__init__(in_dim, (1, 1), parameter, diff)
#         self.N = N
#         self.labels = labels

class TensorAdd(functional):
    def __init__(self, t1, t2, parameter=None, diff=None):
        super(TensorAdd, self).__init__(t1.val.shape, t2.val.shape, parameter, diff)
        self.diff           = np.zeros(self.in_dim)
        self.ahead          = mtr.my_tensor(np.zeros(self.out_dim))
        self.ahead.operator = self
        self.back_num       = 2
        self.back.append(t1)
        self.back.append(t2)

    def fval(self):
        t1 = self.back[0]
        t2 = self.back[1]
        self.ahead.val = t1.val + t2.val

    def grad(self, up_diff):
        return [up_diff, up_diff]
    
    def update_diff(self, up_diff):
        for i in range(self.back_num):
            if self.back[i].type == 1:
                self.diff += self.grad(up_diff)[i]

class TensorDot(functional):
    def __init__(self, t1, t2, parameter=None, diff=None):
        super(TensorDot, self).__init__(t1.val.shape, (1, 1), parameter, diff)
        self.diff           = np.zeros(self.in_dim)
        self.ahead          = mtr.my_tensor(np.zeros(self.out_dim))
        self.ahead.operator = self
        self.back_num       = 2
        self.back.append(t1)
        self.back.append(t2)

    def fval(self):
        t1 = self.back[0]
        t2 = self.back[1]
        self.ahead.val = np.array([[np.sum(t1.val * t2.val)]])

    def grad(self, up_diff):
        d = np.squeeze(up_diff)
        return [d * self.back[1].val, d * self.back[0].val]
    
    def update_diff(self, up_diff):
        for i in range(self.back_num):
            if self.back[i].type == 1:
                self.diff += self.grad(up_diff)[i]
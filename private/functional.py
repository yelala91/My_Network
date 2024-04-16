# functional.py

import numpy as np
import my_tensor as mtr

eps = 1e-6

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
        self.ahead          = mtr.my_tensor()
        self.ahead.operator = self
        self.back_num       = 1

    def zero_diff(self):
        self.diff = np.zeros((self.out_dim, self.in_dim+1))

    def fval(self):
        self.back[0].val = self.back[0].val.reshape(self.in_dim, 1)
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
    def __init__(self, in_dim, shape, filter_num=1, parameter=None, diff=None, padding=0, padding_val=0):
        super(Conv2d, self).__init__(in_dim, None, parameter, diff)
        self.shape          = shape
        self.ahead          = mtr.my_tensor()
        self.filter_num     = filter_num
        self.in_ndim        = len(in_dim)
        self.ahead.operator = self
        self.back_num       = 1
        self.padding, self.padding_val = padding, padding_val
        
        if self.in_ndim == 2:
            if filter_num == 1:
                self.parameter = np.random.randn(*shape)
                self.diff      = np.zeros(shape)
            else:
                self.parameter = np.random.randn(filter_num, *shape)
                self.diff      = np.zeros((filter_num, *shape))
        elif self.in_ndim == 3:
            if filter_num == 1:
                self.parameter = np.random.randn(in_dim[0], *shape)
                self.diff      = np.zeros((in_dim[0], *shape))
            else:
                self.parameter = np.random.randn(filter_num, in_dim[0], *shape)
                self.diff      = np.zeros((filter_num, in_dim[0], *shape))

    def zero_diff(self):
        self.diff = np.zeros(self.shape)
        filter_num = self.filter_num
        shape = self.shape
        
        if self.in_ndim == 2:
            if filter_num == 1:
                self.diff = np.zeros(shape)
            else:
                self.diff = np.zeros((filter_num, *shape))
        elif self.in_ndim == 3:
            if filter_num == 1:
                self.diff = np.zeros((self.in_dim[0], *shape))
            else:
                self.diff = np.zeros((filter_num, self.in_dim[0], *shape))

    def fval(self):
        x = self.back[0].val
        in_ndim = self.in_ndim
        kernel  = self.parameter
        filter_num = self.filter_num
        kernel_h, kernel_w  = self.shape
        out_val = None
        if in_ndim == 2:
            in_h, in_w   = self.in_dim
            out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1

            if filter_num == 1:
                out_val = np.zeros((out_h, out_w))
                if self.padding == 0:
                    for i in range(out_h):
                        for j in range(out_w):
                            out_val[i, j] = np.tensordot(x[i:i+kernel_h, j:j+kernel_w], kernel)
            else:
                out_val = np.zeros((filter_num, out_h, out_w))
                if self.padding == 0:
                    for filter_out_val, filter_kernel in zip(out_val, kernel):
                        for i in range(out_h):
                            for j in range(out_w):
                                filter_out_val[i, j] = np.tensordot(x[i:i+kernel_h, j:j+kernel_w], filter_kernel)
        elif in_ndim == 3:
            channel_num, in_h, in_w = self.in_dim
            out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
            
            if filter_num == 1:
                out_val = np.zeros((out_h, out_w))
                if self.padding == 0:
                    for c in range(channel_num):
                        for i in range(out_h):
                            for j in range(out_w):
                                out_val[i, j] += np.tensordot(x[c, i:i+kernel_h, j:j+kernel_w], kernel[c])
            else:
                out_val = np.zeros((filter_num, out_h, out_w))
                if self.padding == 0:
                    for filter_out_val, filter_kernel in zip(out_val, kernel):
                        for c in range(channel_num):
                            for i in range(out_h):
                                for j in range(out_w):
                                    filter_out_val[i, j] += np.tensordot(x[c, i:i+kernel_h, j:j+kernel_w], filter_kernel[c])
        
        self.ahead.val = out_val

    def grad(self, up_diff):
        
        x = self.back[0].val
        in_ndim = self.in_ndim
        kernel  = self.parameter
        filter_num = self.filter_num
        kernel_h, kernel_w  = self.shape
        grad_x = None
        
        if np.ndim(np.squeeze(up_diff)) == 1:
            up_diff = up_diff.T
        
        if in_ndim == 2:
            in_h, in_w   = self.in_dim
            out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1

            if filter_num == 1:
                grad_x = np.zeros((in_h, in_w))
                for i in range(out_h):
                    for j in range(out_w):
                        if (out_h, out_w) == (1, 1):
                            grad_x[i:i+kernel_h, j:j+kernel_w] += (np.squeeze(up_diff) * kernel)
                        else:
                            grad_x[i:i+kernel_h, j:j+kernel_w] += (up_diff[i, j] * kernel)
            else:
                grad_x = np.zeros((in_h, in_w))
                for filter_kernel, filter_up_diff in zip(kernel, up_diff):
                    for i in range(out_h):
                        for j in range(out_w):
                            if (out_h, out_w) == (1, 1):
                                grad_x[i:i+kernel_h, j:j+kernel_w] += (np.squeeze(filter_up_diff) * filter_kernel)
                            else:
                                grad_x[i:i+kernel_h, j:j+kernel_w] += (filter_up_diff[i, j] * filter_kernel)
                            # grad_x[i:i+kernel_h, j:j+kernel_w] += (filter_up_diff[i, j] * filter_kernel)
        elif in_ndim == 3:
            channel_num, in_h, in_w = self.in_dim
            out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
            
            if filter_num == 1:
                grad_x = np.zeros((channel_num, in_h, in_w))
                for c in range(channel_num):
                    for i in range(out_h):
                        for j in range(out_w):
                            if (out_h, out_w) == (1, 1):
                                grad_x[c, i:i+kernel_h, j:j+kernel_w] += (np.squeeze(up_diff) * kernel[c])
                            else:
                                grad_x[c, i:i+kernel_h, j:j+kernel_w] += (up_diff[i, j] * kernel[c])
                            # grad_x[c, i:i+kernel_h, j:j+kernel_w] += (up_diff[i, j] * kernel[c])
            else:
                grad_x = np.zeros((channel_num, in_h, in_w))
                for filter_kernel, filter_up_diff in zip(kernel, up_diff):
                    for c in range(channel_num):
                        for i in range(out_h):
                            for j in range(out_w):
                                if (out_h, out_w) == (1, 1):
                                    # print(up_diff.shape)
                                    # print(kernel.shape)
                                    grad_x[c, i:i+kernel_h, j:j+kernel_w] += (np.squeeze(filter_up_diff) * filter_kernel[c])
                                else:
                                    grad_x[c, i:i+kernel_h, j:j+kernel_w] += (filter_up_diff[i, j] * filter_kernel[c])
                                # grad_x[c, i:i+kernel_h, j:j+kernel_w] += (filter_up_diff[i, j] * filter_kernel[c])
        
        return [grad_x]
    
    def update_diff(self, up_diff):
        # x = self.back[0].val
        # in_h, in_w          = self.in_dim
        # kernel_h, kernel_w  = self.shape
        # out_h, out_w        = in_h - kernel_h + 1, in_w - kernel_w + 1
        
        x = self.back[0].val
        in_ndim = self.in_ndim
        kernel  = self.parameter
        filter_num = self.filter_num
        kernel_h, kernel_w  = self.shape
        if np.ndim(np.squeeze(up_diff)) == 1:
            up_diff = up_diff.T
        
        if in_ndim == 2:
            in_h, in_w   = self.in_dim
            out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
            
            if filter_num == 1:
                for i in range(kernel_h):
                    for j in range(kernel_w):
                        if out_h == 1 and out_w == 1:
                            self.diff[i, j] += (np.squeeze(up_diff)*np.squeeze(x[i:i+out_h, j:j+out_w]))
                        else:
                            self.diff[i, j] += (np.tensordot(up_diff, x[i:i+out_h, j:j+out_w]))
                        # self.diff[i, j] += (np.tensordot(up_diff, x[i:i+out_h, j:out_w]))
            else:
                for filter_diff, filter_up_diff in zip(self.diff, up_diff):
                    for i in range(kernel_h):
                        for j in range(kernel_w):
                            if out_h == 1 and out_w == 1:
                                filter_diff[i, j] += (np.squeeze(filter_up_diff)*np.squeeze(x[i:i+out_h, j:j+out_w]))
                            else:
                                filter_diff[i, j] += (np.tensordot(filter_up_diff, x[i:i+out_h, j:j+out_w]))
                            # filter_diff[i, j] += (np.tensordot(filter_up_diff, x[i:i+out_h, j:j+out_w]))
        elif in_ndim == 3:
            channel_num, in_h, in_w = self.in_dim
            out_h, out_w = in_h - kernel_h + 1, in_w - kernel_w + 1
            
            if filter_num == 1:
                for c in range(channel_num):
                    for i in range(kernel_h):
                        for j in range(kernel_w):
                            if out_h == 1 and out_w == 1:
                                self.diff[c, i, j] += (np.squeeze(up_diff)*np.squeeze(x[c, i:i+out_h, j:j+out_w]))
                            else:
                                self.diff[c, i, j] += (np.tensordot(up_diff, x[c, i:i+out_h, j:j+out_w]))
                            # self.diff[c, i, j] += (np.tensordot(up_diff, x[c, i:i+out_h, j:j+out_w]))
            else:
                for filter_diff, filter_up_diff in zip(self.diff, up_diff):
                    for c in range(channel_num):
                        for i in range(kernel_h):
                            for j in range(kernel_w):
                                # print(up_diff.shape)
                                # print(x.shape)
                                # print(filter_up_diff.shape)
                                # print(out_h)
                                if out_h == 1 and out_w == 1:
                                    filter_diff[c, i, j] += (np.squeeze(filter_up_diff)*np.squeeze(x[c, i:i+out_h, j:j+out_w]))
                                else:
                                    filter_diff[c, i, j] += (np.tensordot(filter_up_diff, x[c, i:i+out_h, j:j+out_w]))
            
                
# Conv3d
# class Conv3d(functional):
#     def __init__(self, in_dim, shape, parameter=None, diff=None, padding=0, padding_val=0):
#         super(Conv3d, self).__init__(in_dim, None, parameter, diff)
#         self.parameter      = np.random.randn(shape)
#         self.shape          = shape
#         self.diff           = np.zeros(shape)
#         self.ahead          = mtr.my_tensor()
#         self.ahead.operator = self
#         self.back_num       = 1

#     def zero_diff(self):
#         self.diff = np.zeros(self.shape)

#     def fval(self):
#         x = self.back[0].val
        
#         kernel = self.parameter
#         in_h, in_w          = self.in_dim
#         kernel_dim, kernel_h, kernel_w  = self.shape
#         out_h, out_w        = in_h - kernel_h + 1, in_w - kernel_w + 1

#         out_val = np.zeros((out_h, out_w))
#         if self.padding == 0:
#             for i in range(out_h):
#                 for j in range(out_w):
#                     out_val[i][j] = np.tensordot(x[i:i+kernel_h, j:j+kernel_w], kernel)
        
#         self.ahead.val = out_val

#     def grad(self, up_diff):
#         in_h, in_w          = self.in_dim
#         kernel_h, kernel_w  = self.shape
#         out_h, out_w        = in_h - kernel_h + 1, in_w - kernel_w + 1
        
#         grad_x = np.zeros((in_h, in_w))
#         kernel = self.parameter
        
#         for i in range(out_h):
#             for j in range(out_w):
#                 grad_x[i:i+kernel_h, j:j+kernel_w] += (up_diff[i, j] * kernel)
        
#         return [grad_x]
    
#     def update_diff(self, up_diff):
#         x = self.back[0].val
#         in_h, in_w          = self.in_dim
#         kernel_h, kernel_w  = self.shape
#         out_h, out_w        = in_h - kernel_h + 1, in_w - kernel_w + 1
        
#         for i in range(kernel_h):
#             for j in range(kernel_w):
#                 self.diff[i, j] += (np.tensordot(up_diff, x[i:i+out_h, j:out_w]))

# Softmax function
class Softmax(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(Softmax, self).__init__(in_dim, out_dim, parameter, diff)
        self.ahead          = mtr.my_tensor()
        self.ahead.operator = self
        self.back_num       = 1

    def fval(self):
        if type(self.in_dim) is int:
            self.back[0].val = self.back[0].val.reshape(self.in_dim, 1)

        x = self.back[0].val
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
        self.ahead          = mtr.my_tensor()
        self.ahead.operator = self
        self.back_num       = 1

    def fval(self):
        if type(self.in_dim) is int:
            self.back[0].val = self.back[0].val.reshape(self.in_dim, 1)
        
        x = self.back[0].val
        self.ahead.val  = x * (x>=0)

    def grad(self, up_diff):
        x      = self.back[0].val
        if type(self.in_dim) is int:
            grad_x = up_diff * (x.T>=0)
            # print(grad_x.shape)
        else:
            grad_x = up_diff * (x>=0)
        
        return [grad_x]

    def update_diff(self, up_diff):
        pass


class Sigmoid(functional):
    def __init__(self, in_dim, out_dim, parameter=None, diff=None):
        super(Sigmoid, self).__init__(in_dim, out_dim, parameter, diff)
        self.ahead          = mtr.my_tensor()
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
        self.ahead          = mtr.my_tensor()
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
        self.ahead          = mtr.my_tensor()
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
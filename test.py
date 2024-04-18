import os
import sys
sys.path.append('.'+ os.sep + 'private')
import private.neural_network as nn
import private.functional as fnl
import private.my_tensor as mtr
import numpy as np
import train as tr
import mnist_reader as mr
# import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

class test_nn(nn.neural_network):
    def __init__(self):
        super().__init__()

        nn.layers(self, [
            fnl.LinFC(784, 10),
            # fnl.ReLU(256, 256),
            # fnl.LinFC(256, 64),
            # fnl.ReLU(64, 64),
            # fnl.LinFC(64, 10),
            fnl.Softmax(10, 10)
        ])
        
        # nn.layers(self, [
        #     fnl.Conv2d(in_dim=(28, 28), shape=(28, 28), filter_num=10),
        #     # fnl.ReLU((18, 18), (18, 18)),
        #     # fnl.Conv2d(in_dim=(18, 18), shape=(18, 18), filter_num=20),
        #     # fnl.ReLU((5, 20, 20), (5, 20, 20)),
        #     # fnl.Conv2d(in_dim=(5, 20, 20), shape=(20, 20), filter_num=10),
        #     # fnl.ReLU((26, 26), (26, 26)),
        #     # fnl.Conv2d(in_dim=(26, 26), shape=(5, 5), filter_num=1),
        #     # fnl.ReLU(( 22, 22), (22, 22)),
        #     # fnl.Conv2d(in_dim=(22, 22), shape=(22, 22), filter_num=20),
        #     fnl.ReLU(20, 20),
        #     fnl.LinFC(20, 10), 
        #     fnl.Softmax(10, 10)
        # ])

epoch = 50
tnn = test_nn()

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'

x_data, y_data = mr.load_mnist(data_path)
x_test_data, y_test_data = mr.load_mnist(data_path, kind='t10k')

# x_data = x_data.reshape(len(x_data), 28, 28)
n, m = len(x_data), 2048
choice = np.random.choice(n, size=m, replace=False)
x_data, y_data = x_data[choice], y_data[choice]

x_data = x_data/255
x_data = x_data - np.mean(x_data, axis=0)
x_data /= np.var(x_data)*np.var(x_data)>1e-6
y_data = y_data
# x_test_data = x_test_data.reshape(len(x_test_data), 28, 28)
x_test_data = x_test_data/255
# x_test_data = x_test_data - np.mean(x_test_data, axis=0)
train_loss, valid_loss = tr.train(tnn, x_data, y_data, 10 , 32, epoch, sigma=0.03, lr=1)
tr.test(tnn, x_test_data, y_test_data)

para = tnn.parameter()

figure, axes = plt.subplots(2, 5)
axes = axes.reshape(10)
for i in range(10):
    axes[i].imshow(para[0][i, 1:].reshape(28, 28), cmap='gray')


plt.show()


# x = np.random.randn(5, 5)
# # x = np.random.randint(0, 5, size=(5, 5))
# # k = np.ones((3, 3))
# k = np.random.randn(3, 3)

# c1 = fnl.Conv2d((5, 5), (3, 3))
# c1.parameter = k
# c1.back.append(mtr.my_tensor(x))
# c1.fval()

# y1 = c1.ahead.val


# x = x.reshape(1, 1, 5, 5)
# k = k.reshape((1, 1, 3, 3))
# x = torch.from_numpy(x).to(torch.double)
# c2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
# c2.weight.data = torch.from_numpy(k)

# y2 = c2(x)
# print(f'x={x.numpy()}')
# print(f'y1={y1}')
# print(f'y2={y2.detach().numpy()}')


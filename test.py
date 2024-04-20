import os
import sys
sys.path.append('.'+ os.sep + 'private')
import private.neural_network as nn
import private.functional as fnl
import private.my_tensor as mtr
import private.train as tr
import numpy as np
import mnist_reader as mr
# import matplotlib as mpl
import matplotlib.pyplot as plt

class test_nn(nn.neural_network):
    def __init__(self):
        super().__init__()

        nn.layers(self, [
            fnl.LinFC(784, 128),
            fnl.ReLU(128, 128),
            fnl.LinFC(128, 32),
            # fnl.ReLU(32, 32),
            fnl.Sigmoid(32, 32),
            fnl.LinFC(32, 10),
            fnl.Softmax(10, 10)
        ])
        
        # nn.layers(self, [
        #     fnl.Conv2d(in_dim=(28, 28), shape=(5, 5)),
        #     fnl.Pooling2d((24, 24), (12, 12)),
        #     fnl.Conv2d(in_dim=(12, 12), shape=(5, 5)),
        #     fnl.Pooling2d((8, 8), (4, 4)),
        #     fnl.Conv2d((4, 4), (4, 4), filter_num=20),
        #     fnl.ReLU(20, 20),
        #     fnl.LinFC(20, 10),
        #     # fnl.ReLU((18, 18), (18, 18)),
        #     # fnl.Conv2d(in_dim=(18, 18), shape=(18, 18), filter_num=20),
        #     # fnl.ReLU((5, 20, 20), (5, 20, 20)),
        #     # fnl.Conv2d(in_dim=(5, 20, 20), shape=(20, 20), filter_num=10),
        #     # fnl.ReLU((26, 26), (26, 26)),
        #     # fnl.Conv2d(in_dim=(26, 26), shape=(5, 5), filter_num=1),
        #     # fnl.ReLU(( 22, 22), (22, 22)),
        #     # fnl.Conv2d(in_dim=(22, 22), shape=(22, 22), filter_num=20),
        #     # fnl.ReLU(20, 20),
        #     # fnl.LinFC(20, 10), 
        #     fnl.Softmax(10, 10)
        # ])

epoch = 160
tnn = test_nn()

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'

x_data, y_data = mr.load_mnist(data_path)
x_test_data, y_test_data = mr.load_mnist(data_path, kind='t10k')

# x_data = x_data.reshape(len(x_data), 28, 28)
n, m = len(x_data), 3072
choice = np.random.choice(n, size=m, replace=False)
x_data, y_data = x_data[choice], y_data[choice]

x_data = x_data/255
# x_test_data = x_test_data.reshape(len(x_test_data), 28, 28)
x_test_data = x_test_data/255
# x_test_data = x_test_data - np.mean(x_test_data, axis=0)


# tnn.load_param('./saves/my_param_one.npz')
train_loss, valid_loss, train_acc, valid_acc = tr.train(tnn, x_data, y_data, 10 , 32, epoch, sigma=0.12, lr=0.5)

test_acc = tr.test(tnn, x_test_data, y_test_data)
print(f'test acurracy: {test_acc*100: .2f}%')

para = tnn.parameter()
tnn.save_param('./saves/my_param_one')

figure, axes = plt.subplots(2, 5)
axes = axes.reshape(10)

for i in range(10):
    feature = np.matmul(para[2][:, 1:], para[1][:, 1:])
    feature = np.matmul(feature, para[0][:, 1:])
    axes[i].imshow(feature[i].reshape(28, 28), cmap='gray')

plt.figure()
plt.plot(train_acc, label='train acc')
plt.plot(valid_acc, label='valid acc')
plt.legend()

plt.figure()
plt.plot(train_loss, label='train loss')
plt.plot(valid_loss, label='valid loss')
plt.legend()


plt.show()



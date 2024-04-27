# demo.py
# 
# ===============================================================
# A simple demo.
# ===============================================================

import os
import sys
sys.path.append('.'+ os.sep + 'private')
import private.neural_network as nn
import private.train as tr
import numpy as np
import matplotlib.pyplot as plt

epoch           = 80
sigma           = 0.05  # the parameter of L2 regularization
batch_size      = 32
learning_rate   = 0.5
capacity        = 2048

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'
x_train_data, y_train_data, x_test_data, y_test_data = tr.data_read(data_path=data_path, capacity=capacity)

tnn = nn.my_nn()
train_loss, valid_loss, train_acc, valid_acc = tr.train(tnn, x_train_data, y_train_data, 10 , batch_size, epoch, sigma=sigma, lr=learning_rate)

test_acc = tr.test(tnn, x_test_data, y_test_data)
print(f'test acurracy: {test_acc*100: .2f}%')

tnn.save_param('./saves/my_param_test')

tr.display(train_loss, train_acc, valid_loss, valid_acc, settings=(epoch, sigma, batch_size, learning_rate))

plt.show()





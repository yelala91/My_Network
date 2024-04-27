import os
import sys
sys.path.append('.'+ os.sep + 'private')
import private.neural_network as nn
import private.train as tr
import numpy as np
import matplotlib.pyplot as plt


compare = 0     # 0: learning rate, 1: sigma

epoch           = 80
sigmas          = [0.04] #, 0.035,0.03,0.025, 0.02, 0.015, 0.01]   # the parameter of L2 regularization
batch_size      = 32
learning_rates  = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06]
acc_table = np.zeros((len(sigmas), len(learning_rates)))

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'
x_train_data, y_train_data, x_test_data, y_test_data = tr.data_read(data_path=data_path, capacity=2048)
i, j = 0, 0
train_loss_list = []; valid_loss_list = []; train_acc_list = []; valid_acc_list = []
for sigma in sigmas:
    j = 0
    for learning_rate in learning_rates:
        # tnn.load_param('./saves/my_param_one_layer.npz')
        tnn = nn.my_nn()
        train_loss, valid_loss, train_acc, valid_acc = tr.train(tnn, x_train_data, y_train_data, 10 , batch_size, epoch, sigma=sigma, lr=learning_rate)

        test_acc = tr.test(tnn, x_test_data, y_test_data)
        print(f'test acurracy: {test_acc*100: .2f}%')

        para = tnn.parameter()
        # tnn.save_param('./saves/my_param_test')
        
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        # acc_table[i, j] = test_acc
        # tr.display(train_loss, train_acc, valid_loss, valid_acc, settings=(epoch, sigma, batch_size, learning_rate))
        j += 1
    i += 1
    
plt.figure()

# plt.imshow(acc_table)
lenth = len(learning_rates) if compare==0 else len(sigmas)

for k in range(lenth):
    if compare == 0:
        label = f'lr={learning_rates[k]}'
    else:
        label = f'sigma={sigmas[k]}'
        
    plt.plot(train_loss_list[k], label=label)

plt.legend()
out_path = './images/learning_rate_compare_train.png' if compare==0 else './images/sigma_compare_train.png'
plt.savefig(out_path)

plt.figure()
for k in range(lenth):
    if compare == 0:
        label = f'lr={learning_rates[k]}'
    else:
        label = f'sigma={sigmas[k]}'
        
    plt.plot(valid_loss_list[k], label=label)
    
plt.legend()
out_path = './images/learning_rate_compare_valid.png' if compare==0 else './images/sigma_compare_valid.png'
plt.savefig(out_path)

plt.figure()
for k in range(lenth):
    if compare == 0:
        label = f'lr={learning_rates[k]}'
    else:
        label = f'sigma={sigmas[k]}'
        
    plt.plot(train_acc_list[k], label=label)
    
plt.legend()
out_path = './images/learning_rate_compare_train_acc.png' if compare==0 else './images/sigma_compare_train_acc.png'
plt.savefig(out_path)

plt.figure()
for k in range(lenth):
    if compare == 0:
        label = f'lr={learning_rates[k]}'
    else:
        label = f'sigma={sigmas[k]}'
        
    plt.plot(valid_acc_list[k], label=label)
    
plt.legend()
out_path = './images/learning_rate_compare_valid_acc.png' if compare==0 else './images/sigma_compare_valid_acc.png'
plt.savefig(out_path)


plt.show()





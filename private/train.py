# train.py
# 
# ===============================================================
# Some functions about mode training and data processing.
# ===============================================================

import neural_network as nn
import numpy as np
from tqdm import tqdm
from mnist_reader import load_mnist
import matplotlib.pyplot as plt

def data_read(data_path, capacity=2048, out_shape=None):
    x_train_data, y_train_data = load_mnist(data_path)
    x_test_data, y_test_data = load_mnist(data_path, kind='t10k')
    
    if out_shape is not None:
        n1, n2 = len(x_train_data), len(x_test_data)
        x_train_data = x_train_data.reshape(n1, *out_shape)
        x_test_data  = x_test_data.reshape(n2, *out_shape)

    n, m = len(x_train_data), capacity
    choice = np.random.choice(n, size=m, replace=False)
    x_train_data, y_train_data = x_train_data[choice], y_train_data[choice]

    x_train_data = x_train_data/255
    x_test_data = x_test_data/255
    
    return x_train_data, y_train_data, x_test_data, y_test_data

def data_split(x_data, y_data, batch_size, rand=0):
    n = len(x_data)
    per = np.random.permutation(range(n))

    batchs = []; k = 0
    x = []; y = []
    for i in range(n):
        if k < batch_size:
            x.append(x_data[per[i]])
            y.append(y_data[per[i]])
            k += 1
        else:
            batchs.append((x, y))
            x = []; y = []
            k = 0; i -= 1
    return batchs

def valid_data(x_data, y_data, rate=0.1):
    n = len(x_data)
    m = int(n * rate)

    choice = np.random.choice(n, size=m, replace=False)
    x_valid_data = x_data[choice]
    y_valid_data = y_data[choice]

    c_choice = list(set(range(n)) - set(choice))
    x_data = x_data[c_choice]
    y_data = y_data[c_choice]

    return x_valid_data, y_valid_data, x_data, y_data

def acurracy_test(model, x_data, y_data):
    model_fun = lambda x: np.argmax(model.fval(x))
    n = len(x_data)
    acc = 0.0
    for x, y in zip(x_data, y_data):
        if model_fun(x) == y:
            acc += 1
    return acc/n

def train(model, x_data, y_data, k_num,  batch_size=32, epoch=150, sigma=0.1, lr=0.1, type=0):
    x_valid_data, y_valid_data, x_data, y_data = valid_data(x_data, y_data)

    # the record of train loss and valid loss
    train_loss = []; valid_loss = []
    train_acc_list = []; valid_acc_list = []
    best = model.parameter(); best_v_acc = 0
    for i in tqdm(range(epoch)):
        loss = 0
        batchs = data_split(x_data, y_data, batch_size)
        for x, y in batchs:
            loss += nn.Loss(model, x, y, k_num, sigma)  # compute loss and gradient at the same time
            model.update(lr, sigma)                     # update parameters of model base on SGD

        v_loss = np.squeeze(nn.Loss(model, x_valid_data, y_valid_data, k_num, sigma))

        train_loss.append(loss[0,0]/len(batchs))
        valid_loss.append(v_loss)

        # compute the acuuracy on valid set
        valid_acc = acurracy_test(model, x_valid_data, y_valid_data)
        train_acc = acurracy_test(model, x_data, y_data)
        print(f' train acc: {train_acc*100:.2f}%, valid acc: {valid_acc*100:.2f}%, paranorm: {norm_param(model.parameter()):.2e}.')
        valid_acc_list.append(valid_acc)
        train_acc_list.append(train_acc)
        if valid_acc >= best_v_acc:
            best_v_acc = valid_acc
            best = model.parameter()

        # modify the learning rate
        if i % int(epoch/8) == 0:
            lr /= 3
            sigma /= 2.5

    model.load_param_from_list(best)
    return train_loss, valid_loss, train_acc_list, valid_acc_list

def test(model, x_data, y_data):
    model_fun = lambda x: np.argmax(model.fval(x))
    n = len(x_data)
    acc = 0.0
    print('start testing.')
    for x, y in tqdm(zip(x_data, y_data)):
        if model_fun(x) == y:
            acc += 1
    return acc/n

def display(train_loss, train_acc, valid_loss, valid_acc, im_shape=None, para=None):
    
    plt.figure()
    plt.plot(train_acc, label='train acc')
    plt.plot(valid_acc, label='valid acc')
    plt.legend()

    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.legend()
    
    if im_shape is not None:
        figure, axes = plt.subplots(2, 5)
        axes = axes.reshape(10)

        for i in range(10):
            # for p in para:
            #     p = p * p<0
            # feature = np.matmul(para[2][:, 1:], para[1][:, 1:]) 
            # feature = np.matmul(feature, para[0][:, 1:])
            axes[i].imshow(para[0][i, 1:].reshape(*im_shape), cmap='gray', interpolation='bicubic')

    plt.show() 

def norm_param(param_list):
    res = 0.0
    for p in param_list:
        res += np.linalg.norm(p)**2
        
    return res**0.5
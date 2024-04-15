# train.py

import sys
sys.path.append('./private')
import private.neural_network as nn
import private.functional as fnl
import private.my_tensor as mtr
import numpy as np
import matplotlib.pyplot as plt

def data_split(x_data, y_data, batch_size, rand=0):
    n = len(x_data)
    per = np.random.permutation(range(n))

    batch = []
    k = 0
    x = []
    y = []
    for i in range(n):
        
        if k < batch_size:
            x.append(x_data[per[i]])
            y.append(y_data[per[i]])
            k += 1
        else:
            batch.append((x, y))
            x = []
            y = []
            k = 0
            i -= 1
    
    return batch

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

def train(model, x_data, y_data, k_num,  batch_size=32, epoch=150, sigma=0.1, lr=0.1, type=0):
    x_valid_data, y_valid_data, x_data, y_data = valid_data(x_data, y_data)

    train_loss = []
    valid_loss = []
    loss = 300
    for i in range(epoch):
        batch = data_split(x_data, y_data, batch_size)
        ba = 0
        for b in batch:
            x, y = b
            loss = nn.Loss(model, x, y, k_num)
            # print(f"loss = {loss}")
            if type == 0:
                model.update(lr, sigma)
            ba += 1
        train_loss.append(loss[0,0])
        valid_loss.append(np.squeeze(nn.Loss(model, x_valid_data, y_valid_data, k_num)))

        if i % int(epoch/3) == 0:
            lr /= 5
        acc = 0
        for j in range(len(x_valid_data)):
            if np.argmax(model.fval(x_valid_data[j]).val) == y_valid_data[j]:
                acc += 1
        acc = acc/len(x_valid_data)
        if acc > 0.78:
            break
        print(f"train done: {(i+1)/epoch*100:.2f}%, the accuracy: {acc*100:.2f}%, the norm of parameter: {norm_param(model.parameter()):.2e}")

    if type == 0:
        plt.plot(train_loss)
        # plt.plot(valid_loss)
        plt.show()
    
def test(model, x_data, y_data):
    acc = 0
    n = len(x_data)
    for i in range(n):
        print(f'test done: {i/n*100}%')
        if np.argmax(model.fval(x_data[i]).val) == y_data[i]:
            acc += 1
    
    print(f'test accuracy: {acc/n*100}%')
    
def norm_param(param_list):
    res = 0.0
    for p in param_list:
        res += np.linalg.norm(p)
        
    return res
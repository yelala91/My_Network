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

    return x_valid_data, y_valid_data

def train(model, x_data, y_data, k_num,  batch_size, epoch, lr=0.1, type=0):
    x_valid_data, y_valid_data = valid_data(x_data, y_data)

    train_loss = []
    valid_loss = []
    loss = 100
    for i in range(epoch):
        batch = data_split(x_data, y_data, batch_size)
        for b in batch:
            x, y = b
            loss = nn.Loss(model, x, y, k_num)
            # print(f"loss = {loss}")
            if type == 0:
                model.update(lr)
                
        train_loss.append(np.squeeze(loss))
        # valid_loss.append(np.squeeze(nn.Loss(model, x_valid_data, y_valid_data, k_num)))
        print(f"train done: {(i+1)/epoch*100}%")

    print(train_loss)
    
    plt.plot(train_loss)
    # plt.plot(valid_loss)
    plt.show()
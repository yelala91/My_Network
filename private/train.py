# train.py
# 
# ==============================================================
# Some functions about mode training and data processing:
# 
# data_split:
#   Split data base on the batch_size.
#   parameter:
#       x_data------the origin training data.
#       y_data------the lable data of x_data.
#       batch_size--the size for each batch.
#       rand--------the random seed.
#
# valid_data:
#   choose some data as valid data.
#   parameter:
#       x_data------the origin training data.
#       y_data------the lable data of x_data.
#       rate--------the The percentage of the validation set.
#
# train:
#   model training.
#   parameter:
#       model-------the model need to be trained.
#       x_data------the train set.
#       y_data------the label set.
#       k_num-------the total number of class.
#       batch_size--the batch size.
#       epoch-------the epoch.
#       sigma-------the parameter of L2 regularization.
#       lr----------the learning rate.
#       type--------the training type.
#
# test:
#   model test.
# 
# norm_param:
#   return the norm of the parameter of model.
# ===============================================================

import neural_network as nn
import numpy as np

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

def train(model, x_data, y_data, k_num,  batch_size=32, epoch=150, sigma=0.1, lr=0.1, type=0):
    x_valid_data, y_valid_data, x_data, y_data = valid_data(x_data, y_data)

    # the record of train loss and valid loss
    train_loss = []; valid_loss = []
    loss = 0
    for i in range(epoch):
        batchs = data_split(x_data, y_data, batch_size)
        for x, y in batchs:
            loss = nn.Loss(model, x, y, k_num)  # compute loss and gradient at the same time
            loss += 0.5*sigma*norm_param(model.parameter())
            model.update(lr, sigma)             # update parameters of model base on SGD

        v_loss = np.squeeze(nn.Loss(model, x_valid_data, y_valid_data, k_num))
        v_loss += 0.5*sigma*norm_param(model.parameter())
        train_loss.append(loss[0,0])
        valid_loss.append(v_loss)

        # modify the learning rate
        if i % int(epoch/3) == 0:
            lr /= 5

        # compute the acuuracy on valid set
        acc = 0.0
        for vx_data, vy_data in zip(x_valid_data, y_valid_data):
            if np.argmax(model.fval(vx_data).val) == vy_data:
                acc += 1
        acc = acc/len(x_valid_data)

        print(f"train done: {(i+1)/epoch*100:.2f}%, the accuracy: {acc*100:.2f}%, the norm of parameter: {norm_param(model.parameter()):.2e}")
        
    return train_loss, valid_loss

def test(model, x_data, y_data):
    acc = 0.0
    for tx_data, ty_data in zip(x_data, y_data):
        if np.argmax(model.fval(tx_data).val) == ty_data:
            acc += 1
    n = len(x_data)
    print(f'test accuracy: {acc/n*100:.2f}%')
    return acc/len(x_data)
    
def norm_param(param_list):
    res = 0.0
    for p in param_list:
        res += np.linalg.norm(p)
        
    return res
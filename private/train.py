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
from tqdm import tqdm

def data_transform(x_data, y_data, out_shape):
    123

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
            # loss += 0.5*sigma*norm_param(model.parameter())**2
            model.update(lr, sigma)             # update parameters of model base on SGD

        v_loss = np.squeeze(nn.Loss(model, x_valid_data, y_valid_data, k_num, sigma))

        # v_loss += 0.5*sigma*norm_param(model.parameter())**2
        train_loss.append(loss[0,0]/len(batchs))
        valid_loss.append(v_loss)

        # compute the acuuracy on valid set
        valid_acc = acurracy_test(model, x_valid_data, y_valid_data)
        train_acc = acurracy_test(model, x_data, y_data)
        print(f' train acc: {train_acc*100:.2f}%, valid acc: {valid_acc*100:.2f}%, paranorm: {norm_param(model.parameter()):.2e}.')
        valid_acc_list.append(valid_acc)
        train_acc_list.append(train_acc)
        if valid_acc >= best_v_acc + 0.02:
            best_v_acc = valid_acc
            best = model.parameter()

        # modify the learning rate
        if i % int(epoch/8) == 0:
            lr /= 4
            sigma /= 6
        # if i ==  int(epoch/8)*3:
        #     lr /= 10
        #     sigma /= 5
        # if i == int(epoch/8)*5:
        #     lr /= 10
        #     sigma /= 10
        # if i == int(epoch/8)*6:
        #     lr /= 5
        #     sigma /= 5
        # if i == int(epoch/8)*7:
        #     lr /= 5
        #     sigma = 0

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
    
def norm_param(param_list):
    res = 0.0
    for p in param_list:
        res += np.linalg.norm(p)
        
    return res
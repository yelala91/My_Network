# test_my_param.py
# 
# ===============================================================
# test the weight.
# ===============================================================

import os
import sys
sys.path.append('.'+ os.sep + 'private')
import private.neural_network as nn
import private.train as tr

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'
x_train_data, y_train_data, x_test_data, y_test_data = tr.data_read(data_path=data_path)

tnn = nn.my_nn()
tnn.load_param('./saves/my_param.npz')

test_acc = tr.test(tnn, x_test_data, y_test_data)
print(f'test acurracy: {test_acc*100: .2f}%')
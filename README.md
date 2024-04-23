# 神经网络与深度学习 作业一

实现了一个简单的神经网络模型训练框架, 包含了模型的自定义, SGD优化算法以及模型的读写等.

## 数据

本次作业的数据采用了 [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 进行一些服饰识别的训练.

可以采用如下的方式读取数据

```python
import os
import sys
sys.path.append('.'+ os.sep + 'private')

data_path = '.'+ os.sep + 'data' + os.sep + 'fashion'
x_train_data, y_train_data, x_test_data, y_test_data = tr.data_read(data_path=data_path, capacity=2048)
```

## 模型的训练

* 直接运行 `demo.py` 可以开始训练一个三层的全连接神经网络.
* 或者采用如下的方式定义一个神经网络
  
```python
import private.functional as fnl

class my_nn(neural_network):
    def __init__(self):
        super().__init__()
        layers(self, [
            fnl.LinFC(784, 128),
            fnl.ReLU(128, 128),
            fnl.LinFC(128, 32),
            fnl.ReLU(32, 32),
            fnl.LinFC(32, 10),
            fnl.Softmax(10, 10)
        ])
```

目前隐藏层仅支持全连接层 LinFC, 激活函数支持 Sigmoid 和 ReLU. 定义并实例化神经网络后设置超参数, 然后调用 `train(...)` 进行训练.

```python
import private.train as tr

epoch           = 80
sigma           = 0.05   # the parameter of L2 regularization
batch_size      = 32
learning_rate   = 0.5

train_loss, valid_loss, train_acc, valid_acc = tr.train(tnn, x_train_data, y_train_data, 10 , batch_size, epoch, sigma=sigma, lr=learning_rate)
```

## 模型的权重保存

完成神经网络的训练后, 对模型直接调用 `save_param()` 即可.

```python
tnn.save_param('./saves/my_param_test')
```

默认保存地址是 `./saves/`.

## 模型权重的读取

用和权重模型的结构一致的模型调用 `load_param()` 即可.

```python
tnn.load_param('./saves/my_param_test.npz')
```

此时权重文件名称要添加后缀 `.npz`.

## 模型的测试

调用 `test(...)` 进行在测试集上的测试, 返回在测试集上的正确率.

```python
test_acc = tr.test(tnn, x_test_data, y_test_data)
```

import os
import numpy as np
import tensorflow as tf

def get_power_of_2(x):
    '''
    判断x是2的几次方，例如x=8，返回3
    '''
    if x == 1:
        return 0
    count = 1
    while True:
        tmp = divmod(x, 2)
        if tmp[1]!=0:
            raise ValueError('目标分数必须为2的整数次方。')
        if tmp[0]==1:
            return count
        x = tmp[0]
        count += 1

def _make_dir(path):
    '''创建文件目录'''
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'create directionary success: {path}')

def get_device():
    '''
    获取当前设备，CPU or GPU
    '''
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        device = "/gpu:0"
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        device = "/cpu:0"
    return device

def action2onehot(x, n):
    '''
    将离散动作转成one_hot格式
    '''
    assert isinstance(x, int)
    y = np.zeros(n, dtype=np.float32)
    y[x] = 1.
    return y

def observation2int(x):
    '''
    params:
        x: 棋盘状态, 比如4*4
        [[0, 2]
        [4, 8]] => [0, 1, 2, 3]
    '''
    x = np.array(x).flatten()
    count = np.zeros_like(x, dtype=np.int32)
    while True:
        x, _ = divmod(x, 2)
        idx = np.where(x!=0)[0]
        if len(idx):
            count[idx] += 1
        else:
            break
    return count

def batchint2onehot(x, n):
    batchsize = x.shape[0]
    y = x.flatten()
    y = int2one_hot(y, n)
    y = y.reshape([batchsize]+[-1]+list(y.shape)[1:])
    return y


def observation2onehot(x, n):
    '''
    params:
        x: 棋盘状态, 比如4*4
        n: one_hot的长度
    return:
        比如棋盘维度为(4,4), n=6, 则输出维度为(4*4, 6, 1)
    '''
    x = np.array(x).flatten()
    count = np.zeros_like(x, dtype=np.int32)
    while True:
        x, _ = divmod(x, 2)
        idx = np.where(x!=0)[0]
        if len(idx):
            count[idx] += 1
        else:
            break
    return updim(int2one_hot(count, n))

def observation2onehot2(x, n):
    '''
    params:
        x: 棋盘状态, 比如4*4
        n: one_hot的长度
    return:
        比如棋盘维度为(4,4), n=6, 则输出维度为(4, 4, 6)
    '''
    x = np.array(x).flatten()
    count = np.zeros_like(x, dtype=np.int32)
    while True:
        x, _ = divmod(x, 2)
        idx = np.where(x!=0)[0]
        if len(idx):
            count[idx] += 1
        else:
            break
    return int2one_hot(count, n).reshape(4, 4, n)

def int2one_hot(x, n):
    '''
    input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 12
    output: [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
    '''
    x = np.asarray(x).flatten()
    if hasattr(x, '__len__'):
        a = np.zeros([len(x), n])
        for i in range(len(x)):
            a[i, x[i]] = 1
    else:
        a = np.zeros(n)
        a[x] = 1
    return a

def updim(x):
    '''
    数据升维，比如将矩阵[2,2]升维至[2,2,1]
    '''
    return np.expand_dims(np.asarray(x), axis=-1)

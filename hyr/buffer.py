import os
import random
import pickle
import numpy as np
from collections import deque

class Buffer:

    def __init__(self, capacity, batchsize):
        '''初始化经验池
        params:
            capacity: 经验池容量
            batchsize: 采样数据的批大小
        '''
        assert isinstance(capacity, int)
        assert isinstance(batchsize, int)

        self.capacity = capacity
        self.batchsize = batchsize
        self.buffer = deque(maxlen=capacity)

    def add(self, data):
        '''
        存储一条数据至经验池
        '''
        self.buffer.append(data)

    def add_batch(self, data):
        '''
        存储一批数据至经验池
        '''
        [self.add(d) for d in data]

    def sample(self, batchsize=None, _zip=True):
        '''
        从经验池中采样批量数据
        params:
            batchsize: 批的大小，如果未指定，就用初始化经验时指定的批大小
        '''
        batchsize = batchsize or self.batchsize
        data = random.choices(self.buffer, k=batchsize)
        if _zip:
            return [np.asarray(e) for e in zip(*data)]
        else:
            return data

    def __repr__(self):
        '''
        print这个类时，打印经验池中的数据
        '''
        return str(self.buffer)

    def __len__(self):
        '''
        使用len()判断该类长度时，返回经验池中已存数据的数量
        '''
        return len(self.buffer)

    def save(self, name='test'):
        '''
        保存经验池至本地
        '''
        with open(f'./buffer_data/{name}.pkl', 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, name='test'):
        '''
        加载本地经验数据至经验池队列
        '''
        path = f'./buffer_data/{name}.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
        
    def extend(self, x):
        self.buffer.extend(x)


if __name__ == "__main__":
    buffer = Buffer(4, 2)
    buffer.add([1, 'a'])
    print(buffer)
    buffer.add_batch(
        [[2, 'b'],[3, 'c'], [4, 'd'], [5, 'e'], [6, 'f'], [7, 'g']]
        )
    print(buffer)
    data = buffer.sample()
    print(data)


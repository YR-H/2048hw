import os
import sys
sys.path.append('..')
from hyr.utils import _make_dir, get_device
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras import Sequential

class BaseNetwork(M):
    '''CNN神经网络'''

    def __init__(self, s_dim, a_dim, out_dir='test', lr=1.0e-3, is_train=False):
        self._device = get_device()
        super().__init__()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        tf.keras.backend.set_floatx('float32')
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr = lr
        self.is_train = is_train
        self.checkpoint_dir = self.base_dir + '/models/' + out_dir
        self.summary_dir = self.base_dir + '/logs/' + out_dir
        _make_dir(self.checkpoint_dir)
        _make_dir(self.summary_dir)
        self._build_net()
        self(I(shape=self.s_dim))
        self._initialize()

    def _build_net(self):
        pass
                            
    def call(self, s):
        pass

    @tf.function
    def train(self, s, a):
        with tf.device(self._device):
            pass

    def _initialize(self):
        '''
        初始化神经网络模型与优化器
        '''
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
        self.checkpoint = tf.train.Checkpoint(model=self)
        self.saver = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=5, checkpoint_name='cp')
        if self.is_train:
            self.writer = tf.summary.create_file_writer(self.summary_dir)

    def save_model(self, step=None):
        '''
        保存模型到本地
        params:
            step: 当前保存的时间步
        '''
        step = step or self.global_step
        self.saver.save(checkpoint_number=step)

    def load_model(self, _dir=None):
        '''
        加载预训练好的模型
        '''
        _dir = _dir or self.checkpoint_dir
        try:
            self.checkpoint.restore(tf.train.latest_checkpoint(_dir))    # 从指定路径导入模型
        except:
            print(f'restore model from {_dir} FAILED.')
            raise Exception(f'restore model from {_dir} FAILED.')
        else:
            print(f'restore model from {tf.train.latest_checkpoint(_dir)} SUCCUESS.')

    def writer_summary(self, global_step, **kargs):
        """
        记录训练数据，并在tensorboard中显示
        """
        self.writer.set_as_default()
        tf.summary.experimental.set_step(global_step)
        for i in [{'tag': 'AGENT/' + key, 'value': kargs[key]} for key in kargs]:
            tf.summary.scalar(i['tag'], i['value'])
        self.writer.flush()

    def load_model_from_file(self, filepath=None):
        '''
        加载预训练好的模型
        '''
        if filepath is None:
            raise ValueError('file path cannot be None.')
        try:
            self.checkpoint.restore(filepath)    # 从指定路径导入模型
        except:
            print(f'restore model from {filepath} FAILED.')
            raise Exception(f'restore model from {filepath} FAILED.')
        else:
            print(f'restore model from {filepath} SUCCUESS.')
    
if __name__ == "__main__":
    pass

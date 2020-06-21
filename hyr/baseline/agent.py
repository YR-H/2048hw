import sys
sys.path.append('../..')
import numpy as np

from hyr.agent_base import BaseAgent
from hyr.baseline.network import Encoder
from hyr.utils import observation2onehot2

class BaseLineAgent(BaseAgent):
    '''
    模仿学习
    '''
    def __init__(self, game, display=None, is_train=False, **kwargs):
        # 初始化继承类
        self.n = kwargs.get('n', 16)
        self.out_dir = kwargs.get('out_dir', 'nicemodel') # pre_train
        self.is_train = is_train
        super().__init__(game, display=None, **kwargs)

    def choose_action(self):
        x = self.net(np.expand_dims(observation2onehot2(self.game.board, self.n), axis=0))
        direction = np.argmax(x.numpy().flatten())
        return direction
    
    def _build_net(self):
        self.net = Encoder(s_dim=(4, 4, self.n), a_dim=4, out_dir=self.out_dir, is_train=self.is_train)
        self.net.load_model()

    def train_net(self, s, a):
        '''训练神经网络'''
        loss = self.net.train(s, a)
        self.net.writer_summary(global_step=self.net.global_step, loss=loss)
        return loss.numpy()
    
    def writer_summary(self, step, **kwargs):
        '''记录训练日志'''
        self.net.writer_summary(global_step=step, **kwargs)

    def save(self, step=None):
        '''保存模型到本地'''
        self.net.save_model(step)

    def load(self, _dir=None):
        '''加载本地预训练模型'''
        self.net.load_model(_dir)
    
    def step_out(self, state):
        '''执行决策，从外部输入棋盘状态'''
        s = np.expand_dims(state, axis=0)
        a = np.argmax(self.net(s).numpy().flatten())
        return a

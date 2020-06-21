import sys
sys.path.append('..')
import numpy as np
from game2048.agents import Agent

class BaseAgent(Agent):
    '''
    模仿学习智能体
    '''
    def __init__(self, game, display=None, **kwargs):
        # 初始化继承类
        super().__init__(game, display)
        if self.game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        # 获取表示棋盘状态的one_hot长度
        self.n = 16
        # 构建神经网络
        self._build_net()

    def step(self):
        '''执行决策'''
        direction = self.choose_action()
        return direction

    def set_game(self, game):
        '''设置新的棋盘'''
        self.game = game

    # 以下函数需要视情况重载
    
    def _build_net(self):
        self.net = lambda x: x

    def choose_action(self):
        return 0

    def train_net(self, s, a):
        '''训练神经网络'''
        pass
    
    def writer_summary(self, step, **kwargs):
        '''记录训练日志'''
        pass

    def save(self, step=None):
        '''保存模型到本地'''
        pass

    def load(self, _dir=None):
        '''加载本地预训练模型'''
        pass

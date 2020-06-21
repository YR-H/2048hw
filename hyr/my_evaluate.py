import sys
sys.path.append('..')
import numpy as np
from hyr.baseline.agent import BaseLineAgent as Agent
from game2048.game import Game

RANDOM = False  # True or False
NUMS = 100

if __name__ == '__main__':
    env_func = lambda : Game(size=4, score_to_win=2048, random=RANDOM)
    agent = Agent(env_func(), is_train=False, out_dir='empty')
    # agent.net.load_model_from_file('./models/nicemodel/cp-1948')  # restore from .ckpt
    agent.net.load_model('./models/nicemodel')  # restore from ckeckpoint
    scores = []

    for i in range(NUMS):
        agent.play()
        score = agent.game.score
        print(f'{i}: {score}')
        scores.append(score)
        agent.set_game(env_func())

    print(f'average {NUMS}: {np.asarray(scores).mean()}')

import ray
import zmq
import time
import sys
sys.path.append('..')
import random
import numpy as np
import pyarrow as pa
import tensorflow as tf

from buffer import Buffer
from utils import observation2onehot2

@ray.remote
class DS_BUFFER:
    def __init__(self):

        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:10555") 

        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:10556") 

        self.buffer = Buffer(capacity=1000000, batchsize=1024)
        print('init buffer success.')

    def send_batch(self):
        batch = self.buffer.sample(_zip=False)
        batch_id = pa.serialize(batch).to_buffer()
        self.push_socket.send(batch_id)

    def recv_data(self):
        new_replay_data_id = False
        try:
            new_replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass

        if new_replay_data_id:
            new_replay_data = pa.deserialize(new_replay_data_id)
            self.buffer.extend(new_replay_data)

    def run(self):
        while True:
            self.recv_data()
            if len(self.buffer) > 100000:
                break
        while True:
            self.recv_data()
            self.send_batch()

@ray.remote
class DS_LEARNER:
    def __init__(self, _dir, load_dir, start_step=0):

        import sys
        sys.path.append('..')
        from game2048.game import Game
        from juzijiang.baseline.agent import BaseLineAgent as Agent 

        context = zmq.Context()
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:10555") 

        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:10557")

        self.buffer = Buffer(capacity=1000000, batchsize=512)
        self.env_func = lambda: Game(size=4) 
        self.agent = Agent(game=self.env_func(), is_train=True, out_dir=_dir) 
        self.agent.load(load_dir)
        self.train_step = start_step
        print('init learner success.')

    def publish_params(self, weights):
        new_params_id = pa.serialize(weights).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_replay_data_(self):
        replay_data_id = False
        try:
            replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            pass
        if replay_data_id:
            replay_data = pa.deserialize(replay_data_id)
            self.buffer.add_batch(replay_data)

    def run(self):
        time.sleep(3)
        while True:
            self.recv_replay_data_()
            if len(self.buffer) > 256:
                print('start training')
                break
        
        while True:
            self.recv_replay_data_()
            replay_data = self.buffer.sample()
            loss = self.agent.train_net(*replay_data)
            self.train_step += 1

            if self.train_step % 200 == 0:
                self.agent.save(self.train_step)
                print(f'params sending...')
                self.publish_params(self.agent.net.get_weights())
                print(f'send {self.train_step} params success.')

@ray.remote
class DS_WORKER:
    def __init__(self):
        import sys
        sys.path.append('..')
        from game2048.expectimax import board_to_move
        self.move_func = board_to_move

        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:10554") 

        time.sleep(1)
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:10556")

        self.cur_score = 2
        self.buffer = [] 
        print('init worker success.')

    def send_replay_data(self):
        replay_data_id = pa.serialize(self.buffer).to_buffer()
        self.push_socket.send(replay_data_id)

    def receive_new_score(self):
        new_score_id = False
        try:
            new_score_id = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_score_id:
            self.cur_score = pa.deserialize(new_score_id)
            return True

    def run(self):
        step = 0
        while True:
            board = np.random.randint(0, self.cur_score, (4, 4))
            direction = self.move_func(board)
            s = observation2onehot2(board, 16)
            self.buffer.append([s, direction])
            self.buffer.add([observation2onehot2(np.rot90(board, 1), 16), (direction+1) % 4])  # -90
            self.buffer.add([observation2onehot2(np.rot90(board, 2), 16), (direction+2) % 4])  # -180
            self.buffer.add([observation2onehot2(np.rot90(board, 3), 16), (direction+3) % 4])  # -270
            self.buffer.add([observation2onehot2(np.fliplr(board), 16), direction if direction % 2 == 1 else (direction+2) % 4])
            self.buffer.add([observation2onehot2(np.flipud(board), 16), direction if direction % 2 == 0 else (direction+2) % 4])
            step += 1
            if step % 1000 == 0:
                self.send_replay_data()
                self.receive_new_score()
                self.buffer = []
                step = 0

@ray.remote
class DS_VERIFIER:

    def __init__(self):
        import sys
        sys.path.append('..')
        from game2048.game import Game
        from juzijiang.baseline.agent import BaseLineAgent as Agent 

        context = zmq.Context()
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:10554") 

        context = zmq.Context()
        self.sub_socket = context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:10557")

        self.env_func = lambda: Game(size=4) 
        self.agent = Agent(game=self.env_func(), is_train=True, out_dir='finetune_verify') 
        self.recv_times = 0

        print('init verifier success.')

    def receive_new_params(self):
        new_params_id = False
        try:
            new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_params_id:
            new_params = pa.deserialize(new_params_id)
            self.agent.net.set_weights(new_params)
            print(f'load {self.recv_times} params success.')

            success_count = 0
            max_score = 0
            while True:
                self.agent.play()
                score = self.agent.game.score
                print(f'{self.recv_times} {success_count} score: ', score)
                self.agent.set_game(Game())
                max(max_score, score)
                if score >= 2048:
                    success_count += 1
                elif score == 1024:
                    success_count += 0.5
                else:
                    break
                if success_count >= 10:
                    my_agent.save(self.recv_times)
                    break
            self.publish_score(np.log2(max_score)+1)
            self.agent.writer_summary(self.recv_times, score=max_score)
            self.recv_times += 1
            return True

    def publish_score(self, score: int):
        new_score_id = pa.serialize(score).to_buffer()
        self.pub_socket.send(new_score_id)

    def run(self):
        while True:
            self.receive_new_params()


class DS:
    def __init__(self, wokers, _dir='test', load_dir='./models/test', start_step=0):
        ray.init()
        self.all_actors = []

        self.workers = [DS_WORKER.remote() for i in range(wokers)]
        self.all_actors += self.workers

        self.learner = DS_LEARNER.remote(_dir, load_dir, start_step)
        self.all_actors += [self.learner]

        self.buffer = DS_BUFFER.remote()
        self.all_actors += [self.buffer]

        self.verifier = DS_VERIFIER.remote()
        self.all_actors += [self.verifier]

    def train(self):
        ray.wait([actor.run.remote() for actor in self.all_actors])

if __name__ == '__main__':
    ds = DS(wokers=2, _dir='new_train', load_dir='./models/nicemodel', start_step=0)
    print('init success')
    ds.train()
    print('train success')

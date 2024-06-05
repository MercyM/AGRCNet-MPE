import threading
import numpy as np


class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()

        self.buffer['o'] = np.empty([self.size, self.args.num_adversaries, self.args.obs_shape[0]])
        self.buffer['u'] = np.empty([self.size, self.args.num_adversaries, self.args.action_shape[0]])
        self.buffer['r'] = np.empty([self.size, self.args.num_adversaries])
        self.buffer['o_next'] = np.empty([self.size, self.args.num_adversaries, self.args.obs_shape[0]])
        self.buffer['done'] = np.empty([self.size, self.args.num_adversaries])
        self.buffer['C'] = np.empty([self.size, self.args.num_adversaries, self.args.num_adversaries])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next, done, C):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验

        with self.lock:
            self.buffer['o'][idxs] = o
            self.buffer['u'][idxs] = u
            self.buffer['r'][idxs] = r
            self.buffer['o_next'][idxs] = o_next
            self.buffer['done'][idxs] = done
            self.buffer['C'][idxs] = C

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

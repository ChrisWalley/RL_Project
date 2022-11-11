import pickle
from collections import deque
import random as rand


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, state):
        """Save a transition"""
        self.memory.append(state)

    def sample(self, batch_size):
        return rand.sample(list(self.memory), batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, filepath):
        with open(filepath, 'wb') as outp:
            pickle.dump(self.memory, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, 'rb') as inp:
            self.memory = pickle.load(inp)

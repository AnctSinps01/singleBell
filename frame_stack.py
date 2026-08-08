import torch
import numpy as np
from collections import deque


class FrameStacker:
    def __init__(self, history_length=4):
        self.history_length = history_length
        self.frames = deque(maxlen=history_length)
        
    def reset(self, initial_state):
        for _ in range(self.history_length):
            self.frames.append(initial_state)
        return self.get_stacked_state()
        
    def push(self, state):
        self.frames.append(state)
        return self.get_stacked_state()
        
    def get_stacked_state(self):
        stacked = np.array(self.frames)
        return torch.FloatTensor(stacked).unsqueeze(0)
import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32).copy()
        next_state = np.asarray(next_state, dtype=np.float32).copy()
        action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        self.buffer.append(
            (state, action, float(reward), next_state, float(done))
        )

    def sample(self, batch_size, device="cpu"):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"cannot sample {batch_size} transitions from "
                f"a buffer containing {len(self.buffer)}"
            )

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.as_tensor(np.stack(state), device=device),
            torch.as_tensor(np.stack(action), device=device),
            torch.as_tensor(
                reward, dtype=torch.float32, device=device
            ).view(-1, 1),
            torch.as_tensor(np.stack(next_state), device=device),
            torch.as_tensor(
                done, dtype=torch.float32, device=device
            ).view(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)

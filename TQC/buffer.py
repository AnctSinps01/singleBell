import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.states = None
        self.actions = None
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = None
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

    def _allocate(self, state, action):
        self.states = np.empty(
            (self.capacity, *state.shape), dtype=np.float32
        )
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(
            (self.capacity, *action.shape), dtype=np.float32
        )

    def push(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.states is None:
            self._allocate(state, action)
        if state.shape != self.states.shape[1:]:
            raise ValueError("state shape changed after buffer allocation")
        if action.shape != self.actions.shape[1:]:
            raise ValueError("action shape changed after buffer allocation")

        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index, 0] = reward
        self.next_states[index] = next_state
        self.dones[index, 0] = done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if batch_size > self.size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from "
                f"a buffer containing {self.size}"
            )

        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )

    def __len__(self):
        return self.size

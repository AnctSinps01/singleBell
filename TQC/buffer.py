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

    def state_dict(self):
        if self.states is None:
            return {
                "capacity": self.capacity,
                "size": 0,
                "position": 0,
                "states": None,
                "actions": None,
                "rewards": None,
                "next_states": None,
                "dones": None,
            }

        stored = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "states": self.states[:stored].copy(),
            "actions": self.actions[:stored].copy(),
            "rewards": self.rewards[:stored].copy(),
            "next_states": self.next_states[:stored].copy(),
            "dones": self.dones[:stored].copy(),
        }

    def load_state_dict(self, state):
        required = {
            "capacity",
            "size",
            "position",
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"replay buffer state is missing fields: {sorted(missing)}"
            )

        capacity = int(state["capacity"])
        size = int(state["size"])
        position = int(state["position"])
        if capacity != self.capacity:
            raise ValueError(
                f"replay buffer capacity mismatch: {capacity} != "
                f"{self.capacity}"
            )
        if not 0 <= size <= capacity:
            raise ValueError("invalid replay buffer size")
        if not 0 <= position < capacity:
            raise ValueError("invalid replay buffer position")

        if size == 0:
            if any(state[name] is not None for name in required - {
                "capacity", "size", "position"
            }):
                raise ValueError("empty replay buffer contains transition data")
            self.size = 0
            self.position = 0
            self.states = None
            self.actions = None
            self.next_states = None
            return

        arrays = {
            name: np.asarray(state[name], dtype=np.float32)
            for name in (
                "states",
                "actions",
                "rewards",
                "next_states",
                "dones",
            )
        }
        stored = capacity if size == capacity else size
        if any(array.shape[0] != stored for array in arrays.values()):
            raise ValueError("replay buffer arrays have inconsistent lengths")
        if arrays["states"].shape != arrays["next_states"].shape:
            raise ValueError("state and next-state shapes do not match")
        if arrays["rewards"].shape != (stored, 1):
            raise ValueError("rewards must have shape (stored, 1)")
        if arrays["dones"].shape != (stored, 1):
            raise ValueError("dones must have shape (stored, 1)")

        self._allocate(arrays["states"][0], arrays["actions"][0])
        self.states[:stored] = arrays["states"]
        self.actions[:stored] = arrays["actions"]
        self.rewards[:stored] = arrays["rewards"]
        self.next_states[:stored] = arrays["next_states"]
        self.dones[:stored] = arrays["dones"]
        self.size = size
        self.position = position

    def __len__(self):
        return self.size

"""Versioned replay buffers with explicit random-generator and checkpoint state."""

from collections import deque
import copy
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, *, seed=0, priority_alpha=0.0, priority_epsilon=1e-6):
        if (
            type(capacity) is not int
            or capacity < 1
            or not 0 <= priority_alpha <= 1
            or priority_epsilon <= 0
        ):
            raise ValueError("Invalid replay configuration")
        self.capacity, self.alpha, self.epsilon = capacity, priority_alpha, priority_epsilon
        self.rng = np.random.default_rng(seed)
        self.storage = {}
        self.size = self.cursor = self.insertions = 0
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.versions = np.full(capacity, -1, dtype=np.int64)
        self.max_priority = 1.0

    def add(self, transition, *, priority=None):
        values = {key: torch.as_tensor(value).detach().cpu() for key, value in transition.items()}
        if not values:
            raise ValueError("Empty transition")
        if not self.storage:
            self.storage = {
                key: torch.empty((self.capacity, *value.shape), dtype=value.dtype)
                for key, value in values.items()
            }
        if values.keys() != self.storage.keys() or any(
            value.shape != self.storage[key].shape[1:] or value.dtype != self.storage[key].dtype
            for key, value in values.items()
        ):
            raise ValueError("Replay schema changed")
        priority = self.max_priority if priority is None else float(priority)
        if not np.isfinite(priority) or priority < 0:
            raise ValueError("Invalid replay priority")
        for key, value in values.items():
            self.storage[key][self.cursor].copy_(value)
        self.priorities[self.cursor] = max(priority, self.epsilon)
        self.max_priority = max(self.max_priority, priority)
        self.versions[self.cursor] = self.insertions
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.insertions += 1

    def sample(self, batch_size, *, beta=0.4, device="cpu"):
        if batch_size < 1 or not self.size or not 0 <= beta <= 1:
            raise ValueError("Cannot sample empty replay or invalid importance coefficient")

        probabilities = self.priorities[: self.size] ** self.alpha
        probabilities /= probabilities.sum()
        indices = self.rng.choice(self.size, size=batch_size, replace=True, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= (self.size * probabilities.min()) ** (-beta)
        batch = {key: value[indices].to(device) for key, value in self.storage.items()}
        batch.update(
            replay_indices=torch.from_numpy(indices),
            replay_versions=torch.from_numpy(self.versions[indices].copy()),
            importance_weights=torch.as_tensor(weights, device=device, dtype=torch.float32),
        )
        return batch

    def update_priorities(self, indices, versions, priorities):
        for index, version, priority in zip(indices, versions, priorities, strict=True):
            index, version, priority = int(index), int(version), float(priority)
            if not 0 <= index < self.size or not np.isfinite(priority) or priority < 0:
                raise ValueError("Invalid priority update")

            if self.versions[index] != version:
                continue
            self.priorities[index] = max(priority, self.epsilon)
            self.max_priority = max(self.max_priority, priority)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "size": self.size,
            "cursor": self.cursor,
            "insertions": self.insertions,
            "storage": {key: value[: self.size].clone() for key, value in self.storage.items()},
            "priorities": self.priorities.tolist(),
            "versions": self.versions.tolist(),
            "max_priority": self.max_priority,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if (state["capacity"], state["alpha"], state["epsilon"]) != (
            self.capacity,
            self.alpha,
            self.epsilon,
        ):
            raise ValueError("Replay configuration mismatch")
        if not 0 <= state["size"] <= self.capacity or not 0 <= state["cursor"] < self.capacity:
            raise ValueError("Corrupt replay counters")
        storage = {}
        for key, value in state["storage"].items():
            if value.shape[0] != state["size"]:
                raise ValueError("Replay tensor length mismatch")
            storage[key] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            storage[key][: len(value)].copy_(value)
        self.storage = storage
        self.size, self.cursor, self.insertions = (
            state["size"],
            state["cursor"],
            state["insertions"],
        )
        self.priorities = np.asarray(state["priorities"], dtype=np.float64)
        self.versions = np.asarray(state["versions"], dtype=np.int64)
        if (
            self.priorities.shape != (self.capacity,)
            or self.versions.shape != (self.capacity,)
            or not np.isfinite(self.priorities).all()
        ):
            raise ValueError("Corrupt replay metadata")
        self.max_priority = state["max_priority"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class NStepAccumulator:
    def __init__(self, n=3, gamma=0.99):
        if n < 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid n-step return")
        self.n, self.gamma, self.queue = n, gamma, deque()

    def _emit(self):
        first = self.queue[0]
        reward, discount, last = 0.0, 1.0, first
        for transition in list(self.queue)[: self.n]:
            reward += discount * float(transition["reward"])
            discount *= self.gamma
            last = transition
            if bool(transition["terminated"]):
                discount = 0.0
                break
            if bool(transition["truncated"]):
                break
        result = dict(first)
        result.update(
            reward=reward,
            discount=discount,
            next_observation=last["next_observation"],
            terminated=last["terminated"],
            truncated=last["truncated"],
        )
        self.queue.popleft()
        return result

    def add(self, transition):
        if not all(
            key in transition
            for key in (
                "observation",
                "action",
                "reward",
                "next_observation",
                "terminated",
                "truncated",
            )
        ):
            raise ValueError("Incomplete transition")
        self.queue.append(copy.deepcopy(transition))
        results = []
        if transition["terminated"] or transition["truncated"]:
            while self.queue:
                results.append(self._emit())
        elif len(self.queue) >= self.n:
            results.append(self._emit())
        return results

    def state_dict(self):
        return {"n": self.n, "gamma": self.gamma, "queue": list(self.queue)}

    def load_state_dict(self, state):
        if (state["n"], state["gamma"]) != (self.n, self.gamma):
            raise ValueError("N-step configuration mismatch")
        self.queue = deque(copy.deepcopy(state["queue"]))

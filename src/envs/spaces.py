from gymnasium.spaces import Discrete
import numpy as np


class DiscreteSpace(Discrete):
    def __init__(self, n: int):
        self.n = n
        self.dtype = np.int64

    def sample(self, probs=None):
        if probs is None:
            return np.random.randint(self.n)
        else:
            return np.random.choice(self.n, p=probs)

    def contains(self, x):
        return isinstance(x, int) and 0 <= x < self.n

    def __repr__(self):
        return f"Discrete({self.n})"

import numpy as np
from src.agents.base import RLAgent
from src.envs.spaces import DiscreteSpace
from src.agents.utils import preprocess_observation
from pathlib import Path


class QLearningAgent(RLAgent):
    def __init__(
        self,
        action_space: DiscreteSpace,
        max_state_size: int = 1000,
        lr=0.1,
        gamma=0.99,
    ):
        super().__init__(action_space)
        assert max_state_size is not None, "State size must be provided"
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.max_state_size = max_state_size
        self.q_table = np.zeros((max_state_size, action_space.n))

    def predict_action(self, state):
        return np.argmax(self.q_table[state])

    def process_experience(self, state, action, reward, next_state):
        state = preprocess_observation(state, max_states=self.max_state_size)
        next_state = preprocess_observation(next_state, max_states=self.max_state_size)

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def load(self, path: Path):
        assert Path(path).exists(), "Path does not exist"
        assert path.suffix == ".npy", "Path must be a .npy file"
        self.q_table = np.load(path)

    def save(self, path: Path):
        assert path.suffix == ".npy", "Path must be a .npy file"
        np.save(path, self.q_table)

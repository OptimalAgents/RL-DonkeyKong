from typing import Optional
import numpy as np
from src.agents.base import RLAgent
from src.envs.spaces import DiscreteSpace
from src.utils import index_to_action, action_to_index


class QLearningAgent(RLAgent):
    def __init__(
        self,
        action_space: DiscreteSpace,
        state_size: Optional[int] = None,
        lr=0.1,
        gamma=0.99,
    ):
        super().__init__(action_space)
        assert state_size is not None, "State size must be provided"
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((state_size, action_space.n))

    def predict_action(self, state):
        return np.argmax(self.q_table[state])

    def process_experience(self, state, action, reward, next_state, next_action=None):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

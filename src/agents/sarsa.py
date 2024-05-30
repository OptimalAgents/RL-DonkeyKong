from typing import Optional
import numpy as np
from src.agents.base import RLAgent
from src.agents.utils import preprocess_observation
from src.envs.spaces import DiscreteSpace
from src.utils import index_to_action, action_to_index


class SARSAAgent(RLAgent):
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
        action_index = np.argmax(self.q_table[state])
        return index_to_action[action_index]

    def process_experience(self, state, action, reward, next_state, next_action):
        state = preprocess_observation(state, max_states=self.max_state_size)
        next_state = preprocess_observation(next_state, max_states=self.max_state_size)

        next_action_index = action_to_index[next_action]
        td_target = reward + self.gamma * self.q_table[next_state, next_action_index]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

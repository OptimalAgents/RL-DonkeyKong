from __future__ import annotations

import numpy as np

from src.utils import MARIO_COLOR, find_mario, is_barrel_near, ladder_close, CustomActions, index_to_action, \
    action_to_index

from abc import ABC, abstractmethod


class RLAgent(ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def update_q(self, state, action, reward, next_state, next_action=None):
        pass

    @abstractmethod
    def get_q(self, state, action):
        pass


class SARSAAgent(RLAgent):
    def __init__(self, action_space, state_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(action_space)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_size, len(CustomActions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(CustomActions))  # Exploration
        else:
            action_index = np.argmax(self.q_table[state])  # Exploitation
            return index_to_action[action_index]

    def update_q(self, state, action, reward, next_state, next_action):
        action_index = action_to_index[action]
        next_action_index = action_to_index[next_action]
        td_target = reward + self.gamma * self.q_table[next_state, next_action_index]
        td_error = td_target - self.q_table[state, action_index]
        self.q_table[state, action_index] += self.alpha * td_error

    def get_q(self, state, action):
        action_index = action_to_index[action]
        return self.q_table[state, action_index]


class QLearningAgent(RLAgent):
    def __init__(self, action_space, state_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(action_space)
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_size, len(CustomActions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(CustomActions))  # Exploration
        else:
            action_index = np.argmax(self.q_table[state])  # Exploitation
            return index_to_action[action_index]

    def update_q(self, state, action, reward, next_state, next_action=None):
        action_index = action_to_index[action]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action_index]
        self.q_table[state, action_index] += self.alpha * td_error

    def get_q(self, state, action):
        action_index = action_to_index[action]
        return self.q_table[state, action_index]

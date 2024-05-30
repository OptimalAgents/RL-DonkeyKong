from abc import ABC, abstractmethod

from src.envs.spaces import DiscreteSpace


class RLAgent(ABC):
    def __init__(self, action_space: DiscreteSpace):
        self.action_space = action_space

    @abstractmethod
    def predict_action(self, state):
        """Predict best action based on the current state. No exploration"""
        pass

    @abstractmethod
    def process_experience(self, state, action, reward, next_state, next_action=None):
        """Process SARSA tuple and update the model"""
        pass

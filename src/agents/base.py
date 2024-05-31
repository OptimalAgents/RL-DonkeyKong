from abc import ABC, abstractmethod
from pathlib import Path

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

    @abstractmethod
    def load(self, path: Path):
        """Load the model from the given path"""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save the model to the given path"""
        pass

    def step_end_callback(self, step: int):
        """If a model uses experience recall, use this function as a signal that the step has ended"""
        pass

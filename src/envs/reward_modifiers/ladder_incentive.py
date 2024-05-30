from gymnasium.core import ActType, Env, ObsType
import gymnasium as gym
from src.envs.action_wrappers import ReducedActions

from src.envs.utils import find_mario, is_mario_on_ladder


class IncentiveLadder(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], ladder_reward: int = 10):
        self.prev_observation = None
        self.ladder_reward = ladder_reward
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.prev_observation is None:
            self.prev_observation = observation

        mario_prev = find_mario(self.prev_observation)
        mario = find_mario(observation)
        if (
            is_mario_on_ladder(observation)
            and action == ReducedActions.UP
            and (mario[0] < mario_prev[0])
        ):
            reward += self.ladder_reward

        return observation, reward, terminated, truncated, info

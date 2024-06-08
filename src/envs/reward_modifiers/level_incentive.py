from typing import Any, Dict, Tuple
import gymnasium as gym
from gymnasium.core import WrapperObsType

from src.envs.utils import find_mario
from src.envs.utils import get_level


class LevelIncentive(gym.Wrapper):
    def __init__(self, env: gym.Env, level_reward: int = 200):
        super().__init__(env)
        self.level = 0
        self.level_reward = level_reward

    def update_level(self, mario_pos: Tuple[int, int]):
        mario_y = mario_pos[0]
        prev_level = self.level
        curr_level = get_level(mario_y)
        if curr_level > prev_level:
            self.level = curr_level
            print("Level ", curr_level, mario_pos)
            return self.level_reward
        return 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mario = find_mario(observation)

        # HACK: Skip frames until mario is visible
        if mario[0] > 30:
            reward += self.update_level(mario)
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.level = 0
        return self.env.reset(*args, **kwargs)

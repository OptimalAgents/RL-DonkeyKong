from typing import Any, Dict, Tuple
import gymnasium as gym
from gymnasium.core import WrapperObsType

from src.envs.utils import find_mario

FIRST_LEVEL_Y = 158
INBETWEEN_LEVEL_Y = 27


class LevelIncentive(gym.Wrapper):
    def __init__(self, env: gym.Env, level_reward: int = 200):
        super().__init__(env)
        self.level = 0
        self.level_reward = level_reward

    def update_level(self, mario_pos: Tuple[int, int]):
        mario_y = mario_pos[0]
        prev_level = self.level
        if prev_level == 0 and mario_y <= FIRST_LEVEL_Y:
            prev_level = 1
            print("Level ", prev_level, mario_pos)
        if prev_level > 0:
            if mario_y <= (FIRST_LEVEL_Y - INBETWEEN_LEVEL_Y * prev_level):
                prev_level += 1
        if prev_level > self.level:
            self.level = prev_level
            print("Level ", prev_level, mario_pos)
            return self.level_reward
        return 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mario = find_mario(observation)

        # HACK: Skip frames until mario is visible
        if mario[0] > 30:
            reward += self.update_level(mario)
        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.level = 0
        return super().reset(seed=seed, options=options)

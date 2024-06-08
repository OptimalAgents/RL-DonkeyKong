from typing import Any, Dict, SupportsFloat, Tuple
import gymnasium as gym
from gymnasium.core import ActType, Env, ObsType, WrapperObsType
import numpy as np

from src.envs.utils import MARIO_COLOR


STAR_LIST = [
    (187, 61),
    (187, 74),
    (187, 89),
    (183, 102),
    (174, 110),
    (160, 110),
    (160, 99),
    (160, 85),
    (131, 81),
    (131, 89),
    (104, 89),
    (104, 82),
    (104, 73),
    (77, 71),
    (75, 87),
    (75, 100),
    (67, 110),
    (50, 110),
    (50, 97),
    (50, 85),
    (43, 78),
    (25, 76),
]


class MagicStarsIncentive(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], star_reward: int = 20):
        super().__init__(env)
        self.stars_mask = self.create_stars_mask()
        self.star_reward = star_reward

    def create_stars_mask(self):
        # HACK: This is hardcoded env size
        mask = np.zeros((210, 160))
        for star in STAR_LIST:
            mask[star[0], star[1]] = 1
        return mask

    def step(
        self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
        collected_stars = np.where(np.logical_and(mario_mask, self.stars_mask))
        if len(collected_stars[0]) > 0:
            reward += self.star_reward
        self.stars_mask[collected_stars] = 0
        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.stars_mask = self.create_stars_mask()
        return super().reset(seed=seed, options=options)

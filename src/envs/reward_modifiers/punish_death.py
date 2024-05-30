from typing import Any, Dict, SupportsFloat, Tuple
import gymnasium as gym
from gymnasium.core import ActType, Env, ObsType


class PunishDeath(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], death_reward: int = -200):
        super().__init__(env)
        self.death_reward = death_reward

    def step(
        self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward += self.death_reward
        return state, reward, terminated, truncated, info

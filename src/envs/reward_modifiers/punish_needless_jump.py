from typing import Any, Dict, SupportsFloat, Tuple
import gymnasium as gym
from gymnasium.core import ActType, Env, ObsType
from src.envs.action_wrappers import ReducedActions

from src.envs.utils import is_barrel_near, is_mario_on_ladder


class PunishNeedlessJump(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], needless_jump_reward: int = -20):
        super().__init__(env)
        self.prev_observation = None
        self.needless_jump_reward = needless_jump_reward

    def step(
        self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)

        if self.prev_observation is None:
            self.prev_observation = state

        if (
            action == ReducedActions.JUMP
            and not is_mario_on_ladder(self.prev_observation)
            and not is_barrel_near(self.prev_observation)
        ):  # Jump
            reward += self.needless_jump_reward

        return state, reward, terminated, truncated, info

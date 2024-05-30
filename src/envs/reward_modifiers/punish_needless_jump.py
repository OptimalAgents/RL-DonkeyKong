from typing import Any, Dict, SupportsFloat, Tuple
import gymnasium as gym
from gymnasium.core import ActType, Env, ObsType
from src.envs.action_wrappers import ReducedActions

from src.envs.utils import (
    find_barrels,
    find_mario,
    is_barrel_near_mario,
    is_mario_on_ladder,
)


class PunishNeedlessJump(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], needless_jump_reward: int = -20):
        super().__init__(env)
        self.needless_jump_reward = needless_jump_reward

    def step(
        self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)

        mario = find_mario(state)
        barrels = find_barrels(state)

        if (
            action
            in (
                ReducedActions.JUMP,
                ReducedActions.JUMP_LEFT,
                ReducedActions.JUMP_RIGHT,
            )
            and not is_mario_on_ladder(mario)
            and not is_barrel_near_mario(mario, barrels)
        ):  # Jump
            reward += self.needless_jump_reward

        return state, reward, terminated, truncated, info

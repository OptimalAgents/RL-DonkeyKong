import numpy as np
from gymnasium.utils.play import play
from src.envs.action_wrappers import ReducedActions
from src.envs.base import build_base_env, convert_to_playable_env
from src.envs.reward_modifiers import (
    ladder_incentive,
    level_incentive,
    punish_death,
    punish_needless_jump,
)
from src.envs.utils import find_mario

KEY_MAPPING: dict[str, int] = {
    # Available for agents
    "w": ReducedActions.UP,
    "s": ReducedActions.DOWN,
    "a": ReducedActions.LEFT,
    "d": ReducedActions.RIGHT,
    " ": ReducedActions.JUMP,
    "d ": ReducedActions.JUMP_RIGHT,
    "a ": ReducedActions.JUMP_LEFT,
    # Only to make game more enjoyable
    "w ": ReducedActions.UP,
    "s ": ReducedActions.DOWN,
    "wd": ReducedActions.RIGHT,
    "wa": ReducedActions.LEFT,
    "sd": ReducedActions.RIGHT,
    "sa": ReducedActions.LEFT,
    "ws": ReducedActions.NOOP,
    "ad": ReducedActions.NOOP,
}


def main():
    def callback(state_t0, state_t1, action, reward, terminated, truncated, info):
        if state_t0 is None:
            return None
        if isinstance(state_t0, tuple):
            state_t0 = state_t0[0]
        assert isinstance(state_t0, np.ndarray)
        mario = find_mario(state_t0)
        if reward != 0:
            print(reward, mario)
        if terminated:
            print("terminated")
            return
        if truncated:
            print("truncated")
            return

    env = build_base_env(
        level_incentive=True,
    )
    env = convert_to_playable_env(env)
    play(env, fps=30, zoom=5, keys_to_action=KEY_MAPPING, callback=callback)


if __name__ == "__main__":
    main()

import gymnasium as gym
from src.envs.spaces import DiscreteSpace
from enum import IntEnum


class ConvertDescreteActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = DiscreteSpace(env.action_space.n)


class ReducedActions(IntEnum):
    NOOP = 0
    JUMP = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    JUMP_RIGHT = 6
    JUMP_LEFT = 7


REDUCED_ACTION_MAP = {
    ReducedActions.NOOP: 0,
    ReducedActions.JUMP: 1,
    ReducedActions.UP: 2,
    ReducedActions.RIGHT: 3,
    ReducedActions.LEFT: 4,
    ReducedActions.DOWN: 5,
    ReducedActions.JUMP_RIGHT: 11,
    ReducedActions.JUMP_LEFT: 12,
}


class ReduceActionSpace(gym.Wrapper):
    """Reduce the action space to a smaller set of actions. From 0 to n"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = DiscreteSpace(len(REDUCED_ACTION_MAP))

    def step(self, action):
        mapped_action = REDUCED_ACTION_MAP[action]
        return self.env.step(mapped_action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

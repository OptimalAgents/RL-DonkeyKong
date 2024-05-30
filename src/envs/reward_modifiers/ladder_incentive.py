from gymnasium.core import ActType, Env, ObsType
import gymnasium as gym
from src.envs.action_wrappers import ReducedActions

from src.envs.utils import find_mario, is_mario_on_ladder


class LadderIncentive(gym.Wrapper):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        up_ladder_reward: int = 10,
        down_ladder_reward: int = -15,
    ):
        self.prev_mario = None
        self.up_ladder_reward = up_ladder_reward
        self.down_ladder_reward = down_ladder_reward
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.prev_mario is not None:
            mario = find_mario(observation)
            if is_mario_on_ladder(mario):
                if action == ReducedActions.UP and (mario[0] < self.prev_mario[0]):
                    print("YO")
                    reward += self.up_ladder_reward
                elif action == ReducedActions.DOWN and (mario[0] > self.prev_mario[0]):
                    print("NO")
                    reward += self.down_ladder_reward
        self.prev_mario = find_mario(observation)

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.prev_mario = None
        self.env.reset(*args, **kwargs)

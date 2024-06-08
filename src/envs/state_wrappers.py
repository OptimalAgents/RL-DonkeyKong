import numpy as np
import gymnasium as gym
from stable_baselines3.common.type_aliases import AtariResetReturn, AtariStepReturn


class NormalizeObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.dtype = np.float32

    def observation(self, observation):
        return observation / 255.0


class StateSnapshot(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.snapshot = None

    def reset(self, *args, **kwargs):
        self.snapshot = None
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, termination, truncation, info = self.env.step(action)
        self.snapshot = state
        return state, reward, termination, truncation, info


class BetterEpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    Modified from Stable stable_baselines3 package

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int) -> AtariStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info

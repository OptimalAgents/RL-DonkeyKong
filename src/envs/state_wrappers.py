import numpy as np
import gymnasium as gym


class NormalizeObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space.dtype = np.float32

    def observation(self, observation):
        return observation / 255.0

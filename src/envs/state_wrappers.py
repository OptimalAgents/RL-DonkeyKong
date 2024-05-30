import gymnasium as gym


class NormalizeObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation / 255.0

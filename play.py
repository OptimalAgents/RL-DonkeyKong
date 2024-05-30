import numpy as np
from gymnasium.utils.play import play
from src.env import create_playable_env
from src.utils import find_mario
from PIL import Image


def main():
    def callback(state_t0, state_t1, action, reward, terminated, truncated, info):
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

    env = create_playable_env()
    play(env, fps=30, zoom=5, callback=callback)


if __name__ == "__main__":
    main()

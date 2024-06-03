import os
import pickle
from gymnasium import spaces
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from play_with_recording import GameRecorder
import numpy as np


def main():
    buffer_size = 10000
    observation_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    action_space = spaces.Discrete(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the directory containing saved game data files
    save_dir = 'recorded_games'

    # Load and push data to ReplayBuffer
    replay_buffer = GameRecorder.load_and_push_to_replay_buffer(save_dir, buffer_size, observation_space, action_space,
                                                                device)


if __name__ == "__main__":
    main()

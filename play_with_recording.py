import os
import pickle
from datetime import datetime
import numpy as np
from gymnasium import spaces
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium.utils.play import play
from src.envs.base import build_base_env, convert_to_playable_env
from play import KEY_MAPPING

class GameRecorder:
    def __init__(self):
        self.data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

    def callback(self, state_t0, state_t1, action, reward, terminated, truncated, info):
        if state_t0 is None:
            return None
        if isinstance(state_t0, tuple):
            state_t0 = state_t0[0]
        assert isinstance(state_t0, np.ndarray)

        self.data['states'].append(state_t0)
        self.data['actions'].append(action)
        self.data['rewards'].append(reward)
        self.data['dones'].append(terminated or truncated)

        if terminated or truncated:
            self.save_data()
            self.reset_data()

    def save_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'recorded_games/game_data_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Data saved to {filename}")

    def reset_data(self):
        self.data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }

    @staticmethod
    def load_and_push_to_replay_buffer(file_path, buffer_size, observation_space, action_space, device):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        for i in range(len(data['states']) - 1):
            replay_buffer.add(
                obs=data['states'][i],
                next_obs=data['states'][i + 1],
                action=data['actions'][i],
                reward=data['rewards'][i],
                done=data['dones'][i]
            )

        print(f"Pushed {len(data['states']) - 1} transitions to the replay buffer")
        return replay_buffer


def main():
    recorder = GameRecorder()
    env = build_base_env(
        level_incentive=True,
    )
    env = convert_to_playable_env(env)
    play(env, fps=30, zoom=5, keys_to_action=KEY_MAPPING, callback=recorder.callback)


if __name__ == "__main__":
    main()

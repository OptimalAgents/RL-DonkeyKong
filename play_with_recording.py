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
    def load_and_push_to_replay_buffer(directory, buffer_size, observation_space, action_space, device):
        replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        file_count = 0
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                for i in range(len(data['states']) - 1):
                    action = np.array(data['actions'][i]).reshape((1,))  # Reshape action correctly
                    replay_buffer.add(
                        obs=np.array(data['states'][i]),
                        next_obs=np.array(data['states'][i + 1]),
                        action=action,
                        reward=np.array([data['rewards'][i]]),  # Reward needs to be a numpy array
                        done=np.array([data['dones'][i]]),  # Done flag needs to be a numpy array
                        infos={}  # Add empty dictionary for infos
                    )
                file_count += 1

        print(f"Pushed data from {file_count} files to the replay buffer")

        # Print non-zero actions from the last pushed game to see if everything went well
        last_actions = data['actions']
        if last_actions is not None:
            non_zero_actions = [action for action in last_actions if action != 0]
            print("Non-zero actions from the last pushed game:", non_zero_actions)

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

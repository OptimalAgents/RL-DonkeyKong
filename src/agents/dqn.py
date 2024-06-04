from pathlib import Path
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
import sys

from src.agents.base import RLAgent
from src.envs.spaces import DiscreteSpace


class QNetwork(nn.Module):
    def __init__(self, action_space: DiscreteSpace):
        super().__init__()
        self.network1 = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.network2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n),
        )

    def forward(self, x):
        x = x.reshape(-1, 4, 84, 84)
        if x.dtype != torch.float32:
            print(x.dtype)
            print(x.shape)
        # Shape of x: (batch, 4, 84, 84)
        y = self.network1(x)
        return self.network2(y)


class DeepQLearningAgent(RLAgent):
    def __init__(
            self,
            action_space: DiscreteSpace,
            observation_space: gym.Space = None,
            buffer_size: int = 100_000,
            train_start: int = 1_000,
            train_frequency: int = 4,
            target_train_frequency: int = 1_000,
            batch_size: int = 32,
            lr: float = 1e-4,
            gamma: float = 0.99,
            tau: float = 1,
    ):
        super().__init__(action_space)
        assert observation_space is not None, "Observation space must be provided"

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(self.device)

        self.q_network = QNetwork(action_space).to(self.device)
        self.target_q_network = QNetwork(action_space).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            observation_space,
            action_space,
            self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        self.train_start = train_start
        self.train_frequency = train_frequency
        self.target_train_frequency = target_train_frequency

    def predict_action(self, state):
        state = np.array(state, dtype=np.float32)
        q_values = self.q_network(torch.Tensor(state).to(self.device))
        action = torch.argmax(q_values).cpu().numpy()
        return int(action)

    def process_experience(
            self,
            state,
            action,
            reward,
            next_state,
            done=None,
            info=None,
    ):
        self.replay_buffer.add(state, next_state, action, reward, done, [info])

    def load_human_data(self, human_data):
        for experience in human_data:
            state, action, reward, next_state, done = experience
            self.replay_buffer.add(state, next_state, action, reward, done, [{}])

    def step_end_callback(self, step: int):
        super().step_end_callback(step)

        if step > self.train_start:
            super().step_end_callback(step)

        if step > self.train_start:
            if step % self.train_frequency == 0:
                data = self.replay_buffer.sample(self.batch_size)

                agent_actions = torch.argmax(self.q_network(data.observations), dim=1)

                matching_actions_mask = agent_actions == data.actions.flatten()
                if matching_actions_mask.sum() == 0:
                    return

                filtered_observations = data.observations[matching_actions_mask]
                filtered_next_observations = data.next_observations[matching_actions_mask]
                filtered_rewards = data.rewards[matching_actions_mask]
                filtered_dones = data.dones[matching_actions_mask]
                filtered_actions = data.actions[matching_actions_mask]

                with torch.no_grad():
                    target_max, _ = self.target_q_network(filtered_next_observations).max(dim=1)
                    td_target = filtered_rewards.flatten() + self.gamma * target_max * (1 - filtered_dones.flatten())

                old_val = self.q_network(filtered_observations).gather(1, filtered_actions.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update target network
            if step % self.target_train_frequency == 0:
                for target_network_param, q_network_param in zip(
                        self.target_q_network.parameters(), self.q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        self.tau * q_network_param.data
                        + (1.0 - self.tau) * target_network_param.data
                    )

    def save(self, path: Path):
        assert path.suffix == ".pth", "Path must be a .pth file"
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: Path):
        assert Path(path).exists(), "Path does not exist"
        assert path.suffix == ".pth", "Path must be a .pth file"
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

    def __repr__(self):
        return "DeepQLearningAgent"

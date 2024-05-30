from cv2 import QT_NEW_BUTTONBAR
import torch
from torch import optim
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from src.env import create_eval_env
from src.methods.dqn import QNetwork
from tqdm import tqdm
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envs = gym.vector.SyncVectorEnv([create_eval_env])
assert isinstance(
    envs.single_action_space, gym.spaces.Discrete
), "only discrete action space is supported"

q_network = QNetwork(envs).to(device)
q_network.load_state_dict(torch.load("./models/modelL1.pth"))
q_network.eval()

# TRY NOT TO MODIFY: start the game
obs, _ = envs.reset(seed=40)  # make it deterministic
terminations = np.ndarray([0])
while not terminations.any():
    # ALGO LOGIC: put action logic here
    q_values = q_network(torch.Tensor(obs).to(device))
    actions = torch.argmax(q_values, dim=1).cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                print(f"episodic_return={info['episode']['r']}")

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

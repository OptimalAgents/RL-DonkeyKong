import torch
from torch import optim
import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from src.env import create_trainable_env
from src.methods.dqn import QNetwork
from tqdm import tqdm
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    TOTAL_TIMESTEPS = 2_000_000
    LEARNING_STARTS = 80_000
    TRAIN_FREQUENCY = 4
    TARGET_TRAIN_FREQUENCY = 1_000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 1

    envs = gym.vector.SyncVectorEnv([create_trainable_env])
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        200_000,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=40)  # make it deterministic
    for global_step in tqdm(range(TOTAL_TIMESTEPS)):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(1, 0.01, 0.1 * TOTAL_TIMESTEPS, global_step)
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > LEARNING_STARTS:
            if global_step % TRAIN_FREQUENCY == 0:
                data = rb.sample(BATCH_SIZE)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + GAMMA * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % TARGET_TRAIN_FREQUENCY == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        TAU * q_network_param.data
                        + (1.0 - TAU) * target_network_param.data
                    )

    model_path = "./models/model.pth"
    torch.save(q_network.state_dict(), model_path)
    print(f"model saved to {model_path}")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import gymnasium as gym


class DQN(torch.nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        # Memory only
        self.input_shape = input_shape  # bs x 128
        self.num_actions = num_actions

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.input_shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_actions),
        )

    def forward(self, x):
        return self.fc(x)


random = torch.tensor(np.random.random((32, 128)), dtype=torch.float32)
model = DQN(random.shape, 18)
print(model(random).shape)


# Training parameters
num_episodes = 100
max_steps_per_episode = 500

# Setting up game
env = gym.make("ALE/DonkeyKong-ram-v5", render_mode="rgb_array", grayscale_obs=True)
observation, info = env.reset()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # for step in range(max_steps_per_episode):
    #
    # # Print episode information
    # print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Play a game using the learned Q-values
state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state, :])
    state, _, done, _ = env.step(action)
    env.render()

# Close the environment
env.close()

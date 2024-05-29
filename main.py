import numpy as np
import gymnasium as gym
from typing import List, Tuple

from gymnasium.utils.play import play

from src.Agents import SARSAAgent
from src.env import create_playable_env, QLearningAgent, train_q_learning_agent, ActionProbabilityModifier, \
    preprocess_observation, IncentiveLadder, train_sarsa_agent
from src.utils import find_mario, CustomActions
from PIL import Image

from gymnasium.wrappers import TransformReward
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv

from src.env import LevelIncentive, PunishDeath, PunishNeedlessJump, IncentiveMagicStars
from src.utils import is_barrel_near, find_barrels, find_mario, ladder_close

BARREL_COLOR = np.array((236, 200, 96))
TORCH_MASK = np.zeros((210, 160))
TORCH_MASK[68:75, 40:42] = 1

MARIO_COLOR = np.array((200, 72, 72))


def create_wrapped_env() -> gym.Env:
    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")  # Ustaw render_mode na "human"
    env.action_space = gym.spaces.Discrete(len(CustomActions))  # Only 4 custom actions
    env = EpisodicLifeEnv(env)
    env = LevelIncentive(env)
    env = IncentiveLadder(env)
    env = PunishDeath(env)
    env = PunishNeedlessJump(env)
    env = IncentiveMagicStars(env)
    env = TransformReward(env, lambda r: r / 10.0)
    return env


def main(algorithm="q_learning"):
    env = create_wrapped_env()
    state_shape = 1000  # Reduced state space size

    if algorithm == "q_learning":
        agent = QLearningAgent(env.action_space, state_shape)
        train_agent = train_q_learning_agent
    elif algorithm == "sarsa":
        agent = SARSAAgent(env.action_space, state_shape)
        train_agent = train_sarsa_agent
    else:
        raise ValueError("Unknown algorithm. Use 'q_learning' or 'sarsa'.")

    num_evolutions = 10
    episodes_per_evolution = 100

    for evolution in range(num_evolutions):
        print(f"Starting evolution {evolution + 1}/{num_evolutions}")
        # Train the agent with the chosen algorithm
        trained_agent = train_agent(env, agent, num_episodes=episodes_per_evolution)

        # Evaluate the agent after each evolution
        print(f"Evaluating agent after evolution {evolution + 1}/{num_evolutions}")
        observation, info = env.reset()
        total_reward = 0
        max_steps_per_episode = 1000

        for step in range(max_steps_per_episode):
            state = preprocess_observation(observation)
            action = trained_agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()  # Render each step to display the game

            if terminated or truncated:
                break

        print(f"Total reward after evolution {evolution + 1}: {total_reward}")


if __name__ == "__main__":
    main("q_learning")

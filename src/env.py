from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.envs.utils import get_level
from src.agents import RLAgent
from src.envs.utils import find_mario
from src.envs import ReducedActions


class ActionProbabilityModifier(gym.Wrapper):
    def __init__(self, env: gym.Env, agent: RLAgent):
        super().__init__(env)
        self.agent = agent
        self.x = 0
        self.episode_counter = 0
        self.custom_actions = [
            ReducedActions.NOOP,
            ReducedActions.RIGHT,
            ReducedActions.LEFT,
            ReducedActions.UP,
            ReducedActions.JUMP,
        ]

    def update_x(self):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0 and self.x < 5:
            self.x += 1

    def choose_action(self, state, observation):
        mario_y = find_mario(observation)[0]
        level = get_level(mario_y)

        rand_val = np.random.rand()
        prob_best_action = (90 + self.x) / 100.0

        if rand_val < prob_best_action:
            return self.agent.choose_action(state)  # Best action
        else:
            rand_val = np.random.rand()
            if level == 0:  # Ground level
                if rand_val < 3 / 7:
                    return CustomActions.RIGHT  # Move right
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [
                        action
                        for action in self.custom_actions
                        if action not in [CustomActions.RIGHT, CustomActions.UP]
                    ]
                    return np.random.choice(possible_actions)
            elif level == 1:  # First level
                if rand_val < 3 / 7:
                    return CustomActions.LEFT  # Move left
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [
                        action
                        for action in self.custom_actions
                        if action not in [CustomActions.LEFT, CustomActions.UP]
                    ]
                    return np.random.choice(possible_actions)
            elif level == 2:  # Second level
                if rand_val < 3 / 7:
                    return CustomActions.RIGHT  # Move right
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [
                        action
                        for action in self.custom_actions
                        if action not in [CustomActions.RIGHT, CustomActions.UP]
                    ]
                    return np.random.choice(possible_actions)
            elif level == 3:  # Third level
                if rand_val < 3 / 7:
                    return CustomActions.LEFT  # Move left
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [
                        action
                        for action in self.custom_actions
                        if action not in [CustomActions.LEFT, CustomActions.UP]
                    ]
                    return np.random.choice(possible_actions)
            elif level == 4:  # Fourth level
                if rand_val < 3 / 7:
                    return CustomActions.RIGHT  # Move right
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [
                        action
                        for action in self.custom_actions
                        if action not in [CustomActions.RIGHT, CustomActions.UP]
                    ]
                    return np.random.choice(possible_actions)
            else:  # Any other level
                if rand_val < 3 / 7:
                    return self.agent.action_space.sample()
                else:
                    return CustomActions.NOOP  # No operation

    def step(self, action=None):
        if action is None:
            observation = self.env.render()
            state = preprocess_observation(observation)
            action = self.choose_action(state, observation)

        observation, reward, terminated, truncated, info = self.env.step(action)
        state = preprocess_observation(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, *args, seed: int | None = None, options: dict | None = None):
        self.update_x()  # Update x on each reset
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info


def train_q_learning_agent(env, agent, num_episodes=1000, max_steps_per_episode=1000):
    env = ActionProbabilityModifier(
        env, agent
    )  # Use the modified environment for training

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_observation(observation)
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = env.choose_action(
                state, observation
            )  # Use the modified action selection
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(next_observation)
            agent.update_q(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            env.render()  # Render each step to display the game

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    return agent


def train_sarsa_agent(env, agent, num_episodes=1000, max_steps_per_episode=1000):
    env = ActionProbabilityModifier(
        env, agent
    )  # Use the modified environment for training

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_observation(observation)
        action = env.choose_action(state, observation)  # Initial action
        total_reward = 0

        for step in range(max_steps_per_episode):
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(next_observation)
            next_action = env.choose_action(next_state, next_observation)  # Next action
            agent.update_q(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
            total_reward += reward

            env.render()  # Render each step to display the game

            if terminated or truncated:
                break

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    return agent

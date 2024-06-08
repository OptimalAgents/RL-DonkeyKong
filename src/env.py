from __future__ import annotations

from typing import Any, SupportsFloat, Tuple, Dict, List
import gymnasium as gym
import numpy as np
from pathlib import Path
from datetime import datetime
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.transform_reward import TransformReward
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.core import ActType, Env, ObsType, WrapperObsType
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)

from src.Agents import QLearningAgent, RLAgent
from src.utils import MARIO_COLOR, find_mario, is_barrel_near, ladder_close, CustomActions, index_to_action, \
    action_to_index

FIRST_LEVEL_Y = 158
SECOND_LEVEL_Y = 131
THIRD_LEVEL_Y = 104
FOURTH_LEVEL_Y = 77
FIFTH_LEVEL_Y = 50

INBETWEEN_LEVEL_Y = 27


def preprocess_observation(observation):
    # Downsample by a factor of 8
    downsampled = observation[::8, ::8]
    # Normalize pixel values
    normalized = (downsampled / 255).astype(np.float32)
    # Flatten and hash the values to create a unique state index
    hashed = hash(normalized.tobytes()) % 1000  # Hash and modulo to keep within range
    return hashed


def find_mario(observation: np.ndarray) -> Tuple[int, int]:
    MARIO_COLOR = np.array((200, 72, 72))
    mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
    xs, ys = np.where(mario_mask)
    return int(np.mean(xs)), int(np.mean(ys))


def ladder_close(observation: np.ndarray) -> bool:
    LADDER_MASK = np.zeros((210, 160))
    LADDER_MASK[160:196, 106:112] = 1  # First level
    LADDER_MASK[128:169, 79:84] = 1  # Second level
    LADDER_MASK[132:160, 47:53] = 1  # Second level
    LADDER_MASK[102:140, 87:92] = 1  # Third level
    LADDER_MASK[104:137, 108:112] = 1  # Third level
    LADDER_MASK[74:112, 67:72] = 1  # Fourth level
    LADDER_MASK[71:110, 48:52] = 1  # Fourth level
    LADDER_MASK[45:82, 107:112] = 1  # Fifth level
    LADDER_MASK[20:60, 76:80] = 1  # Sixth level
    LADDER_MASK[20:60, 32:36] = 1  # Sixth level

    mario_coords = find_mario(observation)
    if LADDER_MASK[mario_coords[0], mario_coords[1]] == 1:
        return True
    return False


def find_barrels(observation: np.ndarray) -> List[Tuple[int, int]]:
    BARREL_COLOR = np.array((236, 200, 96))
    TORCH_MASK = np.zeros((210, 160))
    TORCH_MASK[68:75, 40:42] = 1

    barrels = []
    barrels_mask = np.all(observation == BARREL_COLOR, axis=-1)
    barrels_mask = np.logical_and(barrels_mask, np.logical_not(TORCH_MASK))
    while np.any(barrels_mask):
        xs, ys = np.where(barrels_mask)
        x, y = xs[0], ys[0]

        x_start, x_end = max(x - 8, 0), min(x + 8, 210)
        y_start, y_end = max(y - 8, 0), min(y + 8, 160)
        search_space = np.zeros((210, 160))
        search_space[x_start: (x_end + 1), y_start: (y_end + 1)] = 1

        current_barrel = np.logical_and(barrels_mask, search_space)

        xs, ys = np.where(current_barrel)
        barrels.append((int(np.mean(xs)), int(np.mean(ys))))
        barrels_mask = np.logical_and(barrels_mask, np.logical_not(current_barrel))

    return barrels


def is_barrel_near(observation: np.ndarray) -> bool:
    mario = find_mario(observation)
    barrels = find_barrels(observation)
    return any(mario[0] + 8 > barrel[0] > mario[0] - 1 and
               mario[
                   1] + 16 > barrel[1] > mario[1] - 16
               for barrel in barrels)


class ActionProbabilityModifier(gym.Wrapper):
    def __init__(self, env: gym.Env, agent: RLAgent):
        super().__init__(env)
        self.agent = agent
        self.x = 0
        self.episode_counter = 0
        self.custom_actions = [CustomActions.NOOP, CustomActions.RIGHT, CustomActions.LEFT, CustomActions.UP,
                               CustomActions.JUMP]

    def update_x(self):
        self.episode_counter += 1
        if self.episode_counter % 2 == 0 and self.x < 5:
            self.x += 1

    def get_level(self, mario_y: int) -> int:
        if FIRST_LEVEL_Y - INBETWEEN_LEVEL_Y < mario_y <= FIRST_LEVEL_Y:
            return 1
        elif SECOND_LEVEL_Y - INBETWEEN_LEVEL_Y < mario_y <= SECOND_LEVEL_Y:
            return 2
        elif THIRD_LEVEL_Y - INBETWEEN_LEVEL_Y < mario_y <= THIRD_LEVEL_Y:
            return 3
        elif FOURTH_LEVEL_Y - INBETWEEN_LEVEL_Y < mario_y <= FOURTH_LEVEL_Y:
            return 4
        elif FIFTH_LEVEL_Y - INBETWEEN_LEVEL_Y < mario_y <= FIFTH_LEVEL_Y:
            return 5
        return 0  # Ground level

    def choose_action(self, state, observation):
        mario_y = find_mario(observation)[0]
        level = self.get_level(mario_y)

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
                    possible_actions = [action for action in self.custom_actions if
                                        action not in [CustomActions.RIGHT, CustomActions.UP]]
                    return np.random.choice(possible_actions)
            elif level == 1:  # First level
                if rand_val < 3 / 7:
                    return CustomActions.LEFT  # Move left
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [action for action in self.custom_actions if
                                        action not in [CustomActions.LEFT, CustomActions.UP]]
                    return np.random.choice(possible_actions)
            elif level == 2:  # Second level
                if rand_val < 3 / 7:
                    return CustomActions.RIGHT  # Move right
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [action for action in self.custom_actions if
                                        action not in [CustomActions.RIGHT, CustomActions.UP]]
                    return np.random.choice(possible_actions)
            elif level == 3:  # Third level
                if rand_val < 3 / 7:
                    return CustomActions.LEFT  # Move left
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [action for action in self.custom_actions if
                                        action not in [CustomActions.LEFT, CustomActions.UP]]
                    return np.random.choice(possible_actions)
            elif level == 4:  # Fourth level
                if rand_val < 3 / 7:
                    return CustomActions.RIGHT  # Move right
                elif 3 / 7 <= rand_val < 5 / 7:
                    return CustomActions.UP  # Climb ladder (Move up)
                else:
                    possible_actions = [action for action in self.custom_actions if
                                        action not in [CustomActions.RIGHT, CustomActions.UP]]
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.update_x()  # Update x on each reset
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info


def train_q_learning_agent(env, agent, num_episodes=1000, max_steps_per_episode=1000):
    env = ActionProbabilityModifier(env, agent)  # Use the modified environment for training

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = preprocess_observation(observation)
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = env.choose_action(state, observation)  # Use the modified action selection
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
    env = ActionProbabilityModifier(env, agent)  # Use the modified environment for training

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


from src.utils import MARIO_COLOR, find_mario, is_barrel_near, ladder_close


FIRST_LEVEL_Y = 158
INBETWEEN_LEVEL_Y = 27


class LevelIncentive(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.level = 0

    def update_level(self, mario_pos: Tuple[int, int]):
        mario_y = mario_pos[0]
        prev_level = self.level
        if prev_level == 0 and mario_y <= FIRST_LEVEL_Y:
            prev_level = 1
            print("Level ", prev_level, mario_pos)
        if prev_level > 0:
            if mario_y <= (FIRST_LEVEL_Y - INBETWEEN_LEVEL_Y * prev_level):
                prev_level += 1
        if prev_level > self.level:
            self.level = prev_level
            print("Level ", prev_level, mario_pos)
            return 200
        return 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        mario = find_mario(observation)

        # HACK: Skip frames until mario is visible
        if mario[0] > 30:
            reward += self.update_level(mario)
        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.level = 0
        return super().reset(seed=seed, options=options)


class IncentiveLadder(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        self.prev_observation = None
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.prev_observation is None:
            self.prev_observation = observation

        mario_prev = find_mario(self.prev_observation)
        mario = find_mario(observation)
        if (action == 2) and (mario[0] < mario_prev[0]):  # Move on a ladder
            reward += 10

        return observation, reward, terminated, truncated, info


class PunishDeath(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

    def step(
            self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward += -200
        return state, reward, terminated, truncated, info


class PunishNeedlessJump(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        self.prev_observation = None

    def step(
            self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)

        if self.prev_observation is None:
            self.prev_observation = state

        if (action == 1) and not ladder_close(self.prev_observation):  # Jump
            if not is_barrel_near(self.prev_observation):  # If not jump over barrel
                reward += -20

        return state, reward, terminated, truncated, info


STAR_LIST = [
    (187, 61),
    (187, 74),
    (187, 89),
    (183, 102),
    (174, 110),
    (160, 110),
    (160, 99),
    (160, 85),
    (131, 81),
    (131, 89),
    (104, 89),
    (104, 82),
    (104, 73),
    (77, 71),
    (75, 87),
    (75, 100),
    (67, 110),
    (50, 110),
    (50, 97),
    (50, 85),
    (43, 78),
    (25, 76)
]


class IncentiveMagicStars(gym.Wrapper):
    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        self.stars_mask = self.create_stars_mask()

    def create_stars_mask(self):
        mask = np.zeros((210, 160))
        for star in STAR_LIST:
            mask[star[0], star[1]] = 1
        return mask

    def step(
            self, action: ActType
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
        collected_stars = np.where(np.logical_and(mario_mask, self.stars_mask))
        if len(collected_stars[0]) > 0:
            reward += 20
        self.stars_mask[collected_stars] = 0
        return observation, reward, terminated, truncated, info

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.stars_mask = self.create_stars_mask()
        return super().reset(seed=seed, options=options)


def create_base_env() -> gym.Env:
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


def create_playable_env() -> gym.Env:
    env = create_base_env()
    return env


def every_n_episodes(n: int):
    def _every_n_episodes(episode_id):
        return episode_id % n == 0

    return _every_n_episodes


def create_eval_env():
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    videos_dir = Path(f"./videos/eval/{date_str}/")
    videos_dir.mkdir(parents=True, exist_ok=True)

    env = create_base_env()
    env = RecordVideo(env, str(videos_dir), episode_trigger=lambda n: True)
    env = FireResetEnv(env)
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    return env


def create_trainable_env() -> gym.Env:
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    videos_dir = Path(f"./videos/train/{date_str}/")
    videos_dir.mkdir(parents=True, exist_ok=True)

    env = create_base_env()
    env = RecordVideo(env, str(videos_dir), episode_trigger=every_n_episodes(50))
    env = FireResetEnv(env)
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    return env


if __name__ == "__main__":
    env = create_base_env()
    print(env)

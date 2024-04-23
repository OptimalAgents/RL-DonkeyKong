from typing import Any, SupportsFloat
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
from src.utils import MARIO_COLOR, find_mario


FIRST_LEVEL_Y = 158
INBETWEEN_LEVEL_Y = 27


class LevelIncentive(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.level = 0

    def update_level(self, mario_pos: tuple[int, int]):
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
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self.level = 0
        return super().reset(seed=seed, options=options)


class InsentiveLadder(gym.Wrapper):
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
    ) -> tuple[WrapperObsType, float, bool, bool, dict[str, Any]]:
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
    ) -> tuple[WrapperObsType, float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)

        if self.prev_observation is None:
            self.prev_observation = state

        mario_prev = find_mario(self.prev_observation)
        mario = find_mario(state)
        if (action == 1) and (mario[0] < mario_prev[0]):  # Jump
            if reward != 100:  # If not jump over barrel
                reward += -20

        return state, reward, terminated, truncated, info


STAR_LIST = [
    (187, 61),
    (187, 74),
    (187, 89),
    (187, 110),
    (167, 110),
    (167, 110),
    (161, 82),
    (159, 49),
    (148, 82),
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
    ) -> tuple[WrapperObsType, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        mario_mask = np.all(observation == MARIO_COLOR, axis=-1)
        collected_stars = np.where(np.logical_and(mario_mask, self.stars_mask))
        if len(collected_stars[0]) > 0:
            reward += 20
        self.stars_mask[collected_stars] = 0
        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self.stars_mask = self.create_stars_mask()
        return super().reset(seed=seed, options=options)


def create_base_env() -> gym.Env:
    env = gym.make("ALE/DonkeyKong-v5", render_mode="rgb_array")
    env.action_space = gym.spaces.Discrete(10)  # 10 actions and noop
    env = EpisodicLifeEnv(env)
    env = LevelIncentive(env)
    env = PunishDeath(env)
    env = PunishNeedlessJump(env)
    # env = InsentiveLadder(env)
    # env = IncentiveMagicStars(env)
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
    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
    print(env)

from pathlib import Path
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.transform_reward import TransformReward
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from src.envs.action_wrappers import ConvertDescreteActions, ReduceActionSpace
from src.envs.state_wrappers import BetterEpisodicLifeEnv, NormalizeObservations
from src.envs.reward_modifiers import (
    LevelIncentive,
    LadderIncentive,
    MagicStarsIncentive,
    PunishDeath,
    PunishNeedlessJump,
)


def build_base_env(
    display: bool = False,
    level_incentive: bool = False,
    ladder_incentive: bool = False,
    magic_stars_incentive: bool = False,
    punish_death: bool = False,
    punish_needless_jump: bool = False,
) -> gym.Env:
    # Base setup
    render_mode = "human" if display else "rgb_array"
    env = gym.make("ALE/DonkeyKong-v5", render_mode=render_mode)
    env = ConvertDescreteActions(env)
    env = ReduceActionSpace(env)

    if level_incentive:
        env = LevelIncentive(env)
    if ladder_incentive:
        env = LadderIncentive(env)
    if magic_stars_incentive:
        env = MagicStarsIncentive(env)
    if punish_death:
        env = PunishDeath(env)
    if punish_needless_jump:
        env = PunishNeedlessJump(env)

    env = TransformReward(env, lambda r: r / 10.0)
    return env


def convert_to_trainable_env(
    env: gym.Env, observation_shape: tuple[int, int] = (84, 84), frame_stack: int = 4
) -> gym.Env:
    env = NoopResetEnv(env, noop_max=30)
    env = BetterEpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, observation_shape)
    env = GrayScaleObservation(env)
    env = NormalizeObservations(env)
    env = FrameStack(env, frame_stack)
    return env


def convert_to_eval_env(env: gym.Env, video_directory: Path) -> gym.Env:
    assert video_directory.exists()
    env = RecordVideo(env, str(video_directory), episode_trigger=lambda n: True)
    env = convert_to_trainable_env(env)
    return env


def convert_to_playable_env(env: gym.Env) -> gym.Env:
    return env

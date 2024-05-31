from pathlib import Path
import numpy as np
import argparse
import json
from datetime import datetime
from src.action_prob import HeuristicActions, UniformActions
from src.agents.dqn import DeepQLearningAgent
from src.agents.qlearning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from tqdm.auto import tqdm

from src.envs.base import build_base_env, convert_to_eval_env, convert_to_trainable_env


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    parser = argparse.ArgumentParser(prog="PROG")
    parser.add_argument("--agent", default="dqn", type=str)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--total_steps", default=1_000_000, type=int)
    parser.add_argument("--eval_every", default=10_000, type=int)
    parser.add_argument("--training_starts", default=84_000, type=int)
    parser.add_argument("--epsilon_start", default=1, type=float)
    parser.add_argument("--epsilon_end", default=0.01, type=float)
    parser.add_argument("--epsilon_duration", default=0.1, type=float)
    parser.add_argument("--ladder_incentive", action=argparse.BooleanOptionalAction)
    parser.add_argument("--level_incentive", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stars_incentive", action=argparse.BooleanOptionalAction)
    parser.add_argument("--punish_death", action=argparse.BooleanOptionalAction)
    parser.add_argument("--punish_needless_jump", action=argparse.BooleanOptionalAction)
    parser.add_argument("--heuristic_actions", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Create run dir
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"./runs/{date_str}")
    run_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Create env
    env = build_base_env(
        level_incentive=args.level_incentive,
        ladder_incentive=args.ladder_incentive,
        magic_stars_incentive=args.stars_incentive,
        punish_death=args.punish_death,
        punish_needless_jump=args.punish_needless_jump,
    )
    env = convert_to_trainable_env(env)
    eval_env = convert_to_eval_env(env, videos_dir)

    # Create agent
    if args.agent == "dqn":
        agent = DeepQLearningAgent(
            env.action_space,
            observation_space=env.observation_space,
            gamma=args.gamma,
            train_start=args.training_starts,
        )
    elif args.agent == "q_learning":
        agent = QLearningAgent(env.action_space)
    elif args.agent == "sarsa":
        agent = SARSAAgent(env.action_space)
    else:
        raise ValueError("Unknown agent. Use 'dqn', 'q_learning' or 'sarsa'.")

    # Action sampler
    if args.heuristic_actions:
        action_sampler = HeuristicActions()
    else:
        action_sampler = UniformActions()

    # Training loop
    eval_rewards = []
    obs, _ = env.reset(seed=44)
    for global_step in tqdm(range(args.total_steps)):
        epsilon = linear_schedule(
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_duration * args.total_steps,
            global_step,
        )
        if np.random.rand() < epsilon:
            # Epsilon exploration
            action = action_sampler.choose_action(obs)
        else:
            action = agent.predict_action(obs)

        next_obs, reward, termination, truncation, info = env.step(action)
        done = termination or truncation

        real_next_obs = next_obs

        if args.agent == "sarsa":
            next_action = agent.predict_action(real_next_obs)
            agent.process_experience(
                obs,
                np.asarray([action]),
                np.asarray([reward]),
                real_next_obs,
                next_action=np.asarray([next_action]),
            )
        elif args.agent == "q_learning":
            agent.process_experience(
                obs, np.asarray([action]), np.asarray([reward]), real_next_obs
            )
        elif args.agent == "dqn":
            agent.process_experience(
                obs,
                np.asarray([action]),
                np.asarray([reward]),
                real_next_obs,
                done=np.asarray([done]),
                info=info,
            )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        agent.step_end_callback(global_step)
        if done:
            print(f"Training episode finished with reward: {info['episode']['r']}")
            obs, _ = env.reset(seed=44)

        # Eval step
        if global_step % args.eval_every == 0:
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action = agent.predict_action(obs)
                obs, reward, termination, truncation, info = eval_env.step(action)
                done = termination or truncation
            eval_env.reset()
            eval_rewards.append(info["episode"]["r"])

    # Save agent
    if args.agent == "dqn":
        agent.save(run_dir / "agent.pth")
    else:
        agent.save(run_dir / "agent.npy")

    # Save eval rewards
    np.save(run_dir / "eval_rewards.npy", np.array(eval_rewards))

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f)


if __name__ == "__main__":
    main()

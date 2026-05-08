#!/usr/bin/env python3
"""
train_scorer_hpc.py — Training script for scorer agent on NYU HPC.
Phase 2: Scorer trains against frozen trained defender.
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from scorer_env_simple import ScorerEnvSimple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10_000_000,
                        help='Total timesteps to train (default: 10000000)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to scorer checkpoint to resume from (omit .zip)')
    parser.add_argument('--defender', type=str, default='defender_ppo_hpc_final',
                        help='Path to frozen defender model (omit .zip)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_scorer_v2/',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=500_000,
                        help='Save checkpoint every N steps')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = ScorerEnvSimple(defender_model_path=args.defender)

    print("Checking environment...")
    check_env(env, warn=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix='scorer_ppo_hpc'
    )

    if args.checkpoint and os.path.exists(args.checkpoint + '.zip'):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, device='cpu')
        reset_timesteps = False
    else:
        print("Creating new PPO model for scorer Phase 2...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # lowered entropy for v3
            device='cpu',
            tensorboard_log="./tb_logs_scorer_v14/"
        )
        reset_timesteps = True

    print(f"Training scorer for {args.steps:,} steps against frozen defender...")
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=reset_timesteps,
        callback=checkpoint_callback
    )

    print("Saving final scorer model...")
    model.save("scorer_ppo_hpc_v14_final")
    print("Done.")
    env.close()


if __name__ == "__main__":
    main()

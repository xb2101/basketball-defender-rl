#!/usr/bin/env python3
"""
train_hpc.py — Training script for NYU HPC (no ROS2/Gazebo needed).
Uses defender_env_simple.py instead of defender_rl_env.py.
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from defender_env_simple import DefenderEnvSimple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2_000_000,
                        help='Total timesteps to train (default: 2000000)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (omit .zip)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_hpc/',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=100_000,
                        help='Save checkpoint every N steps')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = DefenderEnvSimple()

    print("Checking environment...")
    check_env(env, warn=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix='defender_ppo_hpc'
    )

    if args.checkpoint and os.path.exists(args.checkpoint + '.zip'):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, device='cpu')
        reset_timesteps = False
    else:
        print("Creating new PPO model...")
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
            ent_coef=0.01,
            device='cpu',
            tensorboard_log="./tb_logs_hpc/"
        )
        reset_timesteps = True

    print(f"Training for {args.steps:,} steps...")
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=reset_timesteps,
        callback=checkpoint_callback
    )

    print("Saving final model...")
    model.save("defender_ppo_hpc_v7_final")
    print("Done.")
    env.close()


if __name__ == "__main__":
    main()

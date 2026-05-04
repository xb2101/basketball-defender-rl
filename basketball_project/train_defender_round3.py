#!/usr/bin/env python3
"""
train_defender_round3.py — Round 3 defender training on NYU HPC.
Defender retrains against a frozen trained scorer (PPO v7).
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from defender_env_round3 import DefenderEnvRound3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10_000_000,
                        help='Total timesteps to train (default: 10000000)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to defender checkpoint to resume from (omit .zip)')
    parser.add_argument('--scorer', type=str, default='scorer_ppo_hpc_v7_final',
                        help='Path to frozen scorer model (omit .zip)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints_defender_r3/',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=500_000,
                        help='Save checkpoint every N steps')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = DefenderEnvRound3(scorer_model_path=args.scorer)

    print("Checking environment...")
    check_env(env, warn=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix='defender_ppo_r3'
    )

    if args.checkpoint and os.path.exists(args.checkpoint + '.zip'):
        print(f"Resuming from checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, device='cpu')
        reset_timesteps = False
    else:
        print("Creating new PPO model for defender Round 3...")
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
            tensorboard_log="./tb_logs_defender_r3/"
        )
        reset_timesteps = True

    print(f"Training defender Round 3 for {args.steps:,} steps against frozen scorer v7...")
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=reset_timesteps,
        callback=checkpoint_callback
    )

    print("Saving final defender Round 3 model...")
    model.save("defender_ppo_r3_final")
    print("Done.")
    env.close()


if __name__ == "__main__":
    main()

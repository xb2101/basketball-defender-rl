#!/usr/bin/env python3

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from defender_rl_env import DefenderRLEnv


def main():
    env = DefenderRLEnv()

    try:
        print("Checking environment...")
        check_env(env, warn=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path='./checkpoints_gaussian_v5/',
            name_prefix='defender_ppo'
        )

        if os.path.exists("checkpoints_gaussian_v3/defender_ppo_650000_steps.zip"):
            print("Loading Gaussian_V3 650k checkpoint...")
            model = PPO.load("checkpoints_gaussian_v3/defender_ppo_650000_steps", env=env, device='cpu')
            model.tensorboard_log = "./tb_logs_expGaussian_v5/"  # add this line

        elif os.path.exists("defender_ppo_model.zip"):
            print("Loading existing model...")
            model = PPO.load("defender_ppo_model", env=env, device='cpu')
        else:
            print("Creating new PPO model...")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                learning_rate=3e-4, # higher than before — fresh start can afford it
                n_steps=2048,       # shorter rollout → faster feedback on simple reward
                batch_size=128,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                device='cpu',
                tensorboard_log="./tb_logs_expGaussian_v5/"
            )

        print("Starting training...")
        model.learn(
            total_timesteps=100_000,
            reset_num_timesteps=False,
            callback=checkpoint_callback
        )

        print("Saving model...")
        model.save("defender_ppo_gaussian_v5")
        print("Done.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
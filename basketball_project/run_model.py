#!/usr/bin/env python3

import time
import rclpy
from stable_baselines3 import PPO

from defender_rl_env import DefenderRLEnv


def main():
    print("Loading environment...")
    env = DefenderRLEnv()

    # Enable visual markers
    env.show_markers = True

    print("Waiting for simulator to be ready...")
    timeout = 10.0
    start = time.time()
    while time.time() - start < timeout:
        rclpy.spin_once(env.node, timeout_sec=0.1)
        if all(v is not None for v in [
            env.robot_x, env.robot_y, env.robot_yaw,
            env.scorer_x, env.scorer_y
        ]):
            print("Simulator ready!")
            break
    else:
        print("WARNING: Timed out waiting for simulator data!")

    print("Loading trained model...")
    #model = PPO.load("checkpoints/defender_ppo_729800_steps", device='cpu')
    model = PPO.load("defender_ppo_model.zip", device='cpu')

    try:
        obs, _ = env.reset()
        print("Initial obs:", obs)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Reward: {reward:.3f} | terminated: {terminated}")

            if terminated or truncated:
                print("Episode finished → resetting")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        env.close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import time
import rclpy
from stable_baselines3 import PPO

from defender_rl_env import DefenderRLEnv


def main():
    print("Loading environment...")
    env = DefenderRLEnv()

    # Enable visual markers - if you want to see the green sphere (ideal blocking point)
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

    print("Loading trained hpc v6 model  ...")
    #model = PPO.load("checkpoints_gaussian_v3/defender_ppo_650000_steps", device='cpu')
    #model = PPO.load("defender_ppo_hpc_final", device='cpu')
    model = PPO.load("defender_hpc_v6_final", device='cpu')


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
#!/usr/bin/env python3

import time
import numpy as np

from defender_rl_env import DefenderRLEnv


def main():
    env = DefenderRLEnv()

    try:
        print("\nResetting environment...")
        obs, info = env.reset()

        print("Initial observation:", obs)

        # simple test actions
        test_actions = [
            np.array([0.2, 0.0], dtype=np.float32),
            np.array([0.2, 0.5], dtype=np.float32),
            np.array([0.2, -0.5], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
        ]

        for i, action in enumerate(test_actions, start=1):
            print(f"\n--- Step {i} ---")
            print("Action:", action)

            obs, reward, terminated, truncated, info = env.step(action)

            print("Observation:", obs)
            print("Reward:", reward)
            print("Terminated:", terminated)
            print("Truncated:", truncated)

            if terminated or truncated:
                print("\nEpisode ended → resetting...\n")
                obs, info = env.reset()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        print("\nClosing environment...")
        env.close()


if __name__ == '__main__':
    main()

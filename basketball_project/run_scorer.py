#!/usr/bin/env python3
"""
run_scorer.py — Load and visualize a trained scorer checkpoint in Gazebo.

Usage:
    python3 run_scorer.py --checkpoint scorer_hpc_v1_final
"""

import argparse
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from stable_baselines3 import PPO
import numpy as np


# 3-point arc spawn positions
SCORER_STARTS = [
    (1.0,  0.0),
    (2.5, -3.0),
    (2.5,  3.0),
    (4.5, -3.5),
    (4.5,  3.5),
    (1.0, -3.0),
    (1.0,  3.0),
]

GOAL_X, GOAL_Y = 5.0, 0.0
PAINT_RADIUS = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to scorer model checkpoint (omit .zip)')
    parser.add_argument('--episodes', type=int, default=0,
                        help='Number of episodes to run (0 = run forever)')
    return parser.parse_args()


class ScorerRunner(Node):
    def __init__(self):
        super().__init__('scorer_runner')

        # Scorer robot cmd_vel — uses scorer_robot entity
        self.cmd_pub = self.create_publisher(Twist, '/scorer_robot/cmd_vel', 10)

        self.model_sub = self.create_subscription(
            ModelStates, '/gazebo/model_states',
            self.model_states_callback, 10
        )
        self.set_state_client = self.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state...')

        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_yaw: Optional[float] = None

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def model_states_callback(self, msg):
        try:
            idx = msg.name.index('scorer_robot')
            pose = msg.pose[idx]
            self.robot_x = pose.position.x
            self.robot_y = pose.position.y
            self.robot_yaw = self.quaternion_to_yaw(
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w
            )
        except ValueError:
            pass

    def get_obs(self):
        dist_to_paint = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )
        angle_to_goal = math.atan2(GOAL_Y - self.robot_y, GOAL_X - self.robot_x)
        heading_error = angle_to_goal - self.robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        return np.array([
            self.robot_x, self.robot_y, self.robot_yaw,
            GOAL_X, GOAL_Y,
            dist_to_paint,
            heading_error,
        ], dtype=np.float32)

    def set_pose(self, x, y, z=0.1):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = 'scorer_robot'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.angular.z = 0.0
        req.state = state
        future = self.set_state_client.call_async(req)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def reached_paint(self):
        dist = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )
        return dist <= PAINT_RADIUS

    def wait_for_obs(self, timeout=5.0):
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout:
            if all(v is not None for v in [self.robot_x, self.robot_y, self.robot_yaw]):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False


def main():
    args = parse_args()

    if not rclpy.ok():
        rclpy.init()

    node = ScorerRunner()

    print("Waiting for simulator...")
    node.wait_for_obs()
    print("Simulator ready!")

    print(f"Loading scorer model: {args.checkpoint}")
    model = PPO.load(args.checkpoint, device='cpu')

    episode_count = 0
    step_dt = 0.05

    try:
        # Reset to random start position
        sx, sy = SCORER_STARTS[np.random.randint(len(SCORER_STARTS))]
        node.set_pose(sx, sy)
        time.sleep(0.5)
        node.wait_for_obs()
        obs = node.get_obs()
        print(f"Starting episode 1 from ({sx:.1f}, {sy:.1f})")

        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)

            cmd = Twist()
            cmd.linear.x = float(np.clip(action[0], -0.3, 0.3))
            cmd.angular.z = float(action[1])
            node.cmd_pub.publish(cmd)

            start = time.time()
            while rclpy.ok() and (time.time() - start) < step_dt:
                rclpy.spin_once(node, timeout_sec=0.01)

            obs = node.get_obs()
            step += 1

            reached = node.reached_paint()
            out_of_bounds = (
                node.robot_x < 0.0 or node.robot_x > 5.0 or
                node.robot_y < -4.0 or node.robot_y > 4.0
            )

            if reached or out_of_bounds or step >= 500:
                episode_count += 1
                reason = "reached paint!" if reached else "out of bounds" if out_of_bounds else "max steps"
                print(f"Episode {episode_count} finished ({reason})")

                if args.episodes > 0 and episode_count >= args.episodes:
                    break

                # Reset
                sx, sy = SCORER_STARTS[np.random.randint(len(SCORER_STARTS))]
                node.set_pose(sx, sy)
                node.stop()
                time.sleep(0.5)
                node.wait_for_obs()
                obs = node.get_obs()
                step = 0
                print(f"Starting episode {episode_count + 1} from ({sx:.1f}, {sy:.1f})")

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        node.stop()
        node.destroy_node()


if __name__ == "__main__":
    main()

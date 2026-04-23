#!/usr/bin/env python3

import math
import time
from typing import Optional

import gymnasium as gym
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from gymnasium import spaces


class DefenderRLEnv(gym.Env):
    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            rclpy.init()

        self.node = Node('defender_rl_env')
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.model_sub = self.node.create_subscription(
            ModelStates, '/gazebo/model_states',
            self.model_states_callback, 10
        )
        self.set_state_client = self.node.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for /gazebo/set_entity_state...')

        self.robot_name = 'defender'
        self.scorer_name = 'scorer'

        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_yaw: Optional[float] = None

        self.scorer_x: Optional[float] = None
        self.scorer_y: Optional[float] = None
        self.scorer_vx: float = 0.0   # tracked for velocity obs
        self.scorer_vy: float = 0.0
        self._prev_scorer_x: Optional[float] = None
        self._prev_scorer_y: Optional[float] = None

        # Fixed goal/hoop position
        self.goal_x = 5.0
        self.goal_y = 0.0

        self.step_dt = 0.05
        self.max_steps = 500          # increased from 150
        self.current_step = 0
        self.paint_radius = 0.8
        self.show_markers = False  # enabled in run_model.py

        # FIX 1: Allow backwards movement (low=-0.6 not 0.0)
        self.action_space = spaces.Box(
            low=np.array([-0.6, -2.0], dtype=np.float32),
            high=np.array([0.6, 2.0], dtype=np.float32),
            dtype=np.float32
        )

        # FIX 2: Expanded observation — add goal pos + scorer velocity
        # [robot_x, robot_y, robot_yaw, scorer_x, scorer_y,
        #  scorer_vx, scorer_vy, goal_x, goal_y,
        #  dist_to_block, dist_scorer_to_goal]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -10, -10,
                          -2, -2, -10, -10, 0, 0], dtype=np.float32),
            high=np.array([10, 10, math.pi, 10, 10,
                           2, 2, 10, 10, 20, 20], dtype=np.float32),
            dtype=np.float32
        )

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def model_states_callback(self, msg):
        try:
            idx = msg.name.index(self.robot_name)
            pose = msg.pose[idx]
            self.robot_x = pose.position.x
            self.robot_y = pose.position.y
            self.robot_yaw = self.quaternion_to_yaw(
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w
            )
        except ValueError:
            pass

        try:
            idx = msg.name.index(self.scorer_name)
            pose = msg.pose[idx]
            new_x = pose.position.x
            new_y = pose.position.y

            # Estimate scorer velocity from position delta
            if self._prev_scorer_x is not None:
                self.scorer_vx = (new_x - self._prev_scorer_x) / self.step_dt
                self.scorer_vy = (new_y - self._prev_scorer_y) / self.step_dt

            self._prev_scorer_x = new_x
            self._prev_scorer_y = new_y
            self.scorer_x = new_x
            self.scorer_y = new_y
        except ValueError:
            pass

    def _get_blocking_point(self):
        dx = self.goal_x - self.scorer_x
        dy = self.goal_y - self.scorer_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return self.scorer_x, self.scorer_y
        ux, uy = dx / dist, dy / dist
        target_x = self.scorer_x + 0.3 * ux
        target_y = self.scorer_y + 0.3 * uy
        # Cap to court boundaries only
        target_x = max(0.0, min(5.0, target_x))
        target_y = max(-4.0, min(4.0, target_y))
        return target_x, target_y
    
    def _get_obs(self):
        target_x, target_y = self._get_blocking_point()
        dist_to_block = math.sqrt(
            (self.robot_x - target_x) ** 2 + (self.robot_y - target_y) ** 2
        )
        dist_scorer_to_goal = math.sqrt(
            (self.goal_x - self.scorer_x) ** 2 + (self.goal_y - self.scorer_y) ** 2
        )
        return np.array([
            self.robot_x, self.robot_y, self.robot_yaw,
            self.scorer_x, self.scorer_y,
            self.scorer_vx, self.scorer_vy,
            self.goal_x, self.goal_y,
            dist_to_block,
            dist_scorer_to_goal,
        ], dtype=np.float32)

    def _compute_reward(self):
        target_x, target_y = self._get_blocking_point()
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        dist_to_block = math.sqrt(dx * dx + dy * dy)

        # Proximity reward
        proximity_reward = 2.5 * math.exp(-2.0 * dist_to_block)

        # Interception line reward
        ball_to_goal_x = self.goal_x - self.scorer_x
        ball_to_goal_y = self.goal_y - self.scorer_y
        btg_dist = math.sqrt(ball_to_goal_x**2 + ball_to_goal_y**2)
        if btg_dist > 1e-6:
            t = ((self.robot_x - self.scorer_x) * ball_to_goal_x +
                (self.robot_y - self.scorer_y) * ball_to_goal_y) / (btg_dist ** 2)
            t = max(0.0, min(1.0, t))
            proj_x = self.scorer_x + t * ball_to_goal_x
            proj_y = self.scorer_y + t * ball_to_goal_y
            lateral_dist = math.sqrt((self.robot_x - proj_x)**2 + (self.robot_y - proj_y)**2)
            interception_reward = 5.0 * math.exp(-3.0 * lateral_dist)
        else:
            interception_reward = 0.0

        # Penalty for being too far from scorer — forces following
        dist_to_scorer = math.sqrt(
            (self.robot_x - self.scorer_x)**2 +
            (self.robot_y - self.scorer_y)**2
        )
        if dist_to_scorer > 1.5:
            far_penalty = -0.5 * (dist_to_scorer - 2.5)
        else:
            far_penalty = 0.0

        # Collision avoidance
        if dist_to_scorer < 0.6:
            collision_penalty = -5.0
        else:
            collision_penalty = 0.0

        # Speed penalty when close to block point
        if dist_to_block < 0.5:
            linear_vel = abs(self.last_linear_vel) if hasattr(self, 'last_linear_vel') else 0.0
            speed_penalty = -1.0 * linear_vel
        else:
            speed_penalty = 0.0

        time_penalty = -0.05
        goal_penalty = -15.0 if self._scorer_reached_paint() else 0.0

        return (proximity_reward + interception_reward + far_penalty +
                collision_penalty + speed_penalty + time_penalty + goal_penalty)
    
    def _scorer_reached_paint(self):
        dx = self.goal_x - self.scorer_x
        dy = self.goal_y - self.scorer_y
        return math.sqrt(dx * dx + dy * dy) <= self.paint_radius

    def _set_entity_pose(self, name, x, y, z=0.1):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = name
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
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _wait_for_obs(self, timeout=3.0):
        """Spin until we have a valid observation from both models."""
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout:
            if all(v is not None for v in [
                self.robot_x, self.robot_y, self.robot_yaw,
                self.scorer_x, self.scorer_y
            ]):
                return True
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return False

    def step(self, action):
        self.current_step += 1
        self.last_linear_vel = float(action[0])
        self.last_angular_vel = float(action[1])

        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)

        # Move marker to blocking point (only during testing)
        if self.show_markers:
            bx, by = self._get_blocking_point()
            self._set_entity_pose('block_marker', bx, by, 0.1)

        start = time.time()
        while rclpy.ok() and (time.time() - start) < self.step_dt:
            rclpy.spin_once(self.node, timeout_sec=0.01)


        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._scorer_reached_paint()
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {'scorer_reached_paint': terminated}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._stop_robot()

        # Randomize defender start - full court
        defender_x = np.random.uniform(0.5, 3.0)
        defender_y = np.random.uniform(-3.0, 3.0)

        # Randomize scorer start - full court
        scorer_starts = [
            (1.0, 0.0),
            (1.0, -3.0),
            (3.0, -4.0),
            (1.0, 3.0),
            (3.0, 4.0),
            (5.0, -3.0),
            (5.0, 3.0),
            (2.0, 0.0),
            (3.0, -2.0),
            (3.0, 2.0),
        ]
        sx, sy = scorer_starts[np.random.randint(len(scorer_starts))]

        self._set_entity_pose('defender', defender_x, defender_y, 0.1)
        self._set_entity_pose('scorer', sx, sy, 0.2)

        self._prev_scorer_x = None
        self._prev_scorer_y = None
        self.scorer_vx = 0.0
        self.scorer_vy = 0.0

        self._wait_for_obs()
        return self._get_obs(), {}

    def close(self):
        self._stop_robot()
        self.node.destroy_node()
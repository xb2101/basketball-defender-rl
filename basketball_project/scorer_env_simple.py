#!/usr/bin/env python3
"""
scorer_env_simple.py — Pure Python scorer environment (no ROS2/Gazebo).

Phase 1: Scorer learns to reach the paint from 3-point arc positions.
No defender — just learn court navigation and paint-reaching behavior.

Observation space:
[robot_x, robot_y, robot_yaw,
 goal_x, goal_y,
 dist_to_paint, angle_to_goal]

Action space: [linear_vel, angular_vel] — same as defender
"""

import math
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COURT_X_MIN, COURT_X_MAX = 0.0, 5.0
COURT_Y_MIN, COURT_Y_MAX = -4.0, 4.0
GOAL_X, GOAL_Y = 5.0, 0.0
PAINT_RADIUS = 0.8
STEP_DT = 0.05
MAX_STEPS = 500
MAX_LINEAR = 0.6
MAX_ANGULAR = 2.0

# 3-point arc spawn positions
SCORER_STARTS = [
    (1.0,  0.0),   # top of key
    (2.5, -3.0),   # left wing
    (2.5,  3.0),   # right wing
    (4.5, -3.5),   # left corner
    (4.5,  3.5),   # right corner
    (3.0, -2.0),   # left elbow
    (3.0,  2.0),   # right elbow
]


class ScorerEnvSimple(gym.Env):
    """
    Pure Python scorer environment — Phase 1, no defender.
    Scorer learns to navigate from 3-point arc to the paint.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-MAX_LINEAR, -MAX_ANGULAR], dtype=np.float32),
            high=np.array([MAX_LINEAR, MAX_ANGULAR], dtype=np.float32),
            dtype=np.float32
        )

        # [robot_x, robot_y, robot_yaw, goal_x, goal_y, dist_to_paint, angle_to_goal]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -10, -10, 0, -math.pi], dtype=np.float32),
            high=np.array([10, 10, math.pi, 10, 10, 20, math.pi], dtype=np.float32),
            dtype=np.float32
        )

        self.robot_x = 1.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.current_step = 0
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        dist_to_paint = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )
        angle_to_goal = math.atan2(GOAL_Y - self.robot_y, GOAL_X - self.robot_x)
        # Wrap heading error
        heading_error = angle_to_goal - self.robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        return np.array([
            self.robot_x, self.robot_y, self.robot_yaw,
            GOAL_X, GOAL_Y,
            dist_to_paint,
            heading_error,
        ], dtype=np.float32)

    def _compute_reward(self):
        dist_to_paint = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )

        # Primary: get to the paint
        paint_reward = 5.0 * math.exp(-1.5 * dist_to_paint)

        # Bonus for reaching paint
        paint_bonus = 20.0 if dist_to_paint <= PAINT_RADIUS else 0.0

        # Facing reward — face the goal
        angle_to_goal = math.atan2(GOAL_Y - self.robot_y, GOAL_X - self.robot_x)
        heading_error = angle_to_goal - self.robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        facing_reward = 0.5 * math.cos(heading_error)

        # Smoothness penalty
        smoothness_penalty = -0.3 * abs(self.last_angular_vel)

        # Out of bounds penalty
        out_of_bounds = (
            self.robot_x < COURT_X_MIN or self.robot_x > COURT_X_MAX or
            self.robot_y < COURT_Y_MIN or self.robot_y > COURT_Y_MAX
        )
        bounds_penalty = -10.0 if out_of_bounds else 0.0

        # Time penalty — encourage urgency
        time_penalty = -0.05

        return (paint_reward + paint_bonus + facing_reward +
                smoothness_penalty + bounds_penalty + time_penalty)

    def _reached_paint(self):
        dist = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )
        return dist <= PAINT_RADIUS

    def _apply_action(self, linear_vel, angular_vel):
        self.robot_yaw += angular_vel * STEP_DT
        self.robot_yaw = (self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
        self.robot_x += linear_vel * math.cos(self.robot_yaw) * STEP_DT
        self.robot_y += linear_vel * math.sin(self.robot_yaw) * STEP_DT

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def step(self, action):
        self.current_step += 1
        self.last_linear_vel = float(action[0])
        self.last_angular_vel = float(action[1])

        self._apply_action(self.last_linear_vel, self.last_angular_vel)

        obs = self._get_obs()
        reward = self._compute_reward()

        out_of_bounds = (
            self.robot_x < COURT_X_MIN or self.robot_x > COURT_X_MAX or
            self.robot_y < COURT_Y_MIN or self.robot_y > COURT_Y_MAX
        )
        terminated = self._reached_paint() or out_of_bounds
        truncated = self.current_step >= MAX_STEPS

        return obs, reward, terminated, truncated, {'reached_paint': self._reached_paint()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Spawn at random 3-point arc position
        sx, sy = SCORER_STARTS[np.random.randint(len(SCORER_STARTS))]
        # Add small random offset so it doesn't always start exactly the same
        self.robot_x = sx + np.random.uniform(-0.2, 0.2)
        self.robot_y = sy + np.random.uniform(-0.2, 0.2)
        self.robot_yaw = np.random.uniform(-math.pi, math.pi)

        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0

        return self._get_obs(), {}

    def close(self):
        pass

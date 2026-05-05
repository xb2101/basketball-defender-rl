#!/usr/bin/env python3
"""
defender_env_simple.py — Pure Python defender environment (no ROS2/Gazebo).
Updated with all latest reward changes:
- Interception line reward
- Collision threshold at 0.3m
- Close bonus 10.0 at 0.5m
- Blocking point at 0.6m
- Out of bounds penalty and termination
- Smoothness penalty increased to -3.0 (was -1.5) to reduce oscillation
- Velocity penalty increased to -2.5 (was -1.0) within 0.5m of blocking point 
  to force defender to hold position once it arrives instead of overshooting
"""

import math
import random
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Constants — must match Gazebo setup exactly
# ---------------------------------------------------------------------------
COURT_X_MIN, COURT_X_MAX = 0.0, 5.0
COURT_Y_MIN, COURT_Y_MAX = -4.0, 4.0
GOAL_X, GOAL_Y = 5.0, 0.0
PAINT_RADIUS = 0.8
SCORER_SPEED = 0.3
WAYPOINT_TOL = 0.3
STEP_DT = 0.05
MAX_STEPS = 500
MAX_LINEAR = 0.6
MAX_ANGULAR = 2.0


class ScorerSim:
    """Replicates scorer_controller.py logic exactly."""

    def __init__(self):
        self.x = 1.0
        self.y = 0.0
        self.stage = 0
        self.random_stops_remaining = 0
        self.target_x = 0.0
        self.target_y = 0.0
        self._new_episode()

    def _pick_random_court_position(self):
        return random.uniform(0.5, 3.5), random.uniform(-3.5, 3.5)

    def _new_episode(self):
        self.random_stops_remaining = random.randint(2, 3)
        self.stage = 0
        self.target_x, self.target_y = self._pick_random_court_position()

    def reset(self, x, y):
        self.x = x
        self.y = y
        self._new_episode()

    def step(self):
        if self.stage == 0:
            tx, ty = self.target_x, self.target_y
        else:
            tx, ty = 5.5, 0.0

        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < WAYPOINT_TOL or (self.stage == 1 and self.x >= 4.5):
            if self.stage == 0:
                self.random_stops_remaining -= 1
                if self.random_stops_remaining <= 0:
                    self.stage = 1
                else:
                    self.target_x, self.target_y = self._pick_random_court_position()
            else:
                self._new_episode()
            return False

        ux = dx / dist
        uy = dy / dist
        self.x += SCORER_SPEED * STEP_DT * ux
        self.y += SCORER_SPEED * STEP_DT * uy

        dx_goal = GOAL_X - self.x
        dy_goal = GOAL_Y - self.y
        return math.sqrt(dx_goal**2 + dy_goal**2) <= PAINT_RADIUS


class DefenderEnvSimple(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-MAX_LINEAR, -MAX_ANGULAR], dtype=np.float32),
            high=np.array([MAX_LINEAR, MAX_ANGULAR], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -10, -10,
                          -2, -2, -10, -10, 0, 0], dtype=np.float32),
            high=np.array([10, 10, math.pi, 10, 10,
                           2, 2, 10, 10, 20, 20], dtype=np.float32),
            dtype=np.float32
        )

        self.scorer = ScorerSim()
        self.robot_x = 2.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.scorer_vx = 0.0
        self.scorer_vy = 0.0
        self.current_step = 0
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0

    def _get_blocking_point(self):
        dx = GOAL_X - self.scorer.x
        dy = GOAL_Y - self.scorer.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return self.scorer.x, self.scorer.y
        ux, uy = dx / dist, dy / dist
        tx = self.scorer.x + 0.6 * ux
        ty = self.scorer.y + 0.6 * uy
        tx = max(COURT_X_MIN, min(COURT_X_MAX, tx))
        ty = max(COURT_Y_MIN, min(COURT_Y_MAX, ty))
        return tx, ty

    def _get_obs(self):
        tx, ty = self._get_blocking_point()
        dist_to_block = math.sqrt(
            (self.robot_x - tx) ** 2 + (self.robot_y - ty) ** 2
        )
        dist_scorer_to_goal = math.sqrt(
            (GOAL_X - self.scorer.x) ** 2 + (GOAL_Y - self.scorer.y) ** 2
        )
        return np.array([
            self.robot_x, self.robot_y, self.robot_yaw,
            self.scorer.x, self.scorer.y,
            self.scorer_vx, self.scorer_vy,
            GOAL_X, GOAL_Y,
            dist_to_block,
            dist_scorer_to_goal,
        ], dtype=np.float32)

    def _compute_reward(self):
        tx, ty = self._get_blocking_point()
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        dist_to_block = math.sqrt(dx * dx + dy * dy)

        # Primary signal: Gaussian centered on blocking point
        blocking_reward = 5.0 * math.exp(-1.5 * dist_to_block)

        # Facing bonus
        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - self.robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        facing_reward = 0.3 * math.cos(heading_error)

        # Interception line reward
        ball_to_goal_x = GOAL_X - self.scorer.x
        ball_to_goal_y = GOAL_Y - self.scorer.y
        btg_dist = math.sqrt(ball_to_goal_x**2 + ball_to_goal_y**2)
        if btg_dist > 1e-6:
            t = ((self.robot_x - self.scorer.x) * ball_to_goal_x +
                 (self.robot_y - self.scorer.y) * ball_to_goal_y) / (btg_dist ** 2)
            t = max(0.0, min(0.6, t))
            proj_x = self.scorer.x + t * ball_to_goal_x
            proj_y = self.scorer.y + t * ball_to_goal_y
            lateral_dist = math.sqrt((self.robot_x - proj_x)**2 + (self.robot_y - proj_y)**2)
            interception_reward = 3.0 * math.exp(-3.0 * lateral_dist)
        else:
            interception_reward = 0.0

        # Collision penalty
        dist_to_scorer = math.sqrt(
            (self.robot_x - self.scorer.x) ** 2 +
            (self.robot_y - self.scorer.y) ** 2
        )
        collision_penalty = -5.0 if dist_to_scorer < 0.3 else 0.0

        # Out of bounds penalty
        out_of_bounds = (
            self.robot_x < COURT_X_MIN or self.robot_x > COURT_X_MAX or
            self.robot_y < COURT_Y_MIN or self.robot_y > COURT_Y_MAX
        )
        bounds_penalty = -10.0 if out_of_bounds else 0.0

        # Goal penalty
        goal_penalty = -15.0 if self._scorer_reached_paint() else 0.0

        # Time penalty
        time_penalty = -0.05

        # Smoothness penalty - increased from -1.5 to -3.0 to reduce oscillation
        smoothness_penalty = -3.0 * abs(self.last_angular_vel)

        # Velocity penalty when close to blocking point - increased from -1.0 to -2.5
        # to force defender to hold position once it arrives instead of overshooting
        velocity_penalty = -2.5 * abs(self.last_linear_vel) if dist_to_block < 0.5 else 0.0

        return (blocking_reward + facing_reward +
                interception_reward + collision_penalty +
                bounds_penalty + goal_penalty + time_penalty
                + smoothness_penalty + velocity_penalty)

    def _scorer_reached_paint(self):
        dx = GOAL_X - self.scorer.x
        dy = GOAL_Y - self.scorer.y
        return math.sqrt(dx * dx + dy * dy) <= PAINT_RADIUS

    def _apply_action(self, linear_vel, angular_vel):
        self.robot_yaw += angular_vel * STEP_DT
        self.robot_yaw = (self.robot_yaw + math.pi) % (2 * math.pi) - math.pi
        self.robot_x += linear_vel * math.cos(self.robot_yaw) * STEP_DT
        self.robot_y += linear_vel * math.sin(self.robot_yaw) * STEP_DT

    def step(self, action):
        self.current_step += 1
        self.last_linear_vel = float(action[0])
        self.last_angular_vel = float(action[1])

        self._apply_action(self.last_linear_vel, self.last_angular_vel)

        prev_x, prev_y = self.scorer.x, self.scorer.y
        scored = self.scorer.step()
        self.scorer_vx = (self.scorer.x - prev_x) / STEP_DT
        self.scorer_vy = (self.scorer.y - prev_y) / STEP_DT

        obs = self._get_obs()
        reward = self._compute_reward()

        out_of_bounds = (
            self.robot_x < COURT_X_MIN or self.robot_x > COURT_X_MAX or
            self.robot_y < COURT_Y_MIN or self.robot_y > COURT_Y_MAX
        )
        terminated = scored or out_of_bounds
        truncated = self.current_step >= MAX_STEPS

        return obs, reward, terminated, truncated, {'scorer_reached_paint': scored}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.robot_x = np.random.uniform(2.5, 4.5)
        self.robot_y = np.random.uniform(-3.0, 3.0)
        self.robot_yaw = np.random.uniform(-math.pi, math.pi)

        scorer_starts = [
            (1.0, 0.0), (1.0, -3.0), (3.0, -4.0),
            (1.0, 3.0), (3.0, 4.0), (5.0, -3.0),
            (5.0, 3.0), (2.0, 0.0), (3.0, -2.0), (3.0, 2.0),
        ]
        sx, sy = scorer_starts[np.random.randint(len(scorer_starts))]
        self.scorer.reset(sx, sy)

        self.scorer_vx = 0.0
        self.scorer_vy = 0.0

        return self._get_obs(), {}

    def close(self):
        pass

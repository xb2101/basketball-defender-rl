#!/usr/bin/env python3
"""
scorer_env_simple.py — Pure Python scorer environment (no ROS2/Gazebo).

Phase 2: Scorer trains against frozen trained defender.
- 10 observations including defender position
- Scorer can see defender and learn to navigate around it

Observation space (10 dims):
[scorer_x, scorer_y, scorer_yaw,
 defender_x, defender_y,
 goal_x, goal_y,
 dist_to_paint, dist_to_defender, angle_to_goal]
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COURT_X_MIN, COURT_X_MAX = 0.0, 5.0
COURT_Y_MIN, COURT_Y_MAX = -4.0, 4.0
GOAL_X, GOAL_Y = 5.0, 0.0
PAINT_RADIUS = 1.0
STEP_DT = 0.05
MAX_STEPS = 500
MAX_LINEAR = 0.6
MAX_ANGULAR = 2.0

SCORER_STARTS = [
    (1.0,  0.0),
    (2.5, -3.0),
    (2.5,  3.0),
    (4.5, -3.5),
    (4.5,  3.5),
    (3.0, -2.0),
    (3.0,  2.0),
]


class DefenderSim:
    """Simulates the trained defender using a frozen PPO model."""

    def __init__(self, model_path: str):
        print(f"Loading frozen defender model from {model_path}...")
        self.model = PPO.load(model_path, device='cpu')
        self.x = 3.0
        self.y = 0.0
        self.yaw = 0.0
        self.scorer_vx = 0.0
        self.scorer_vy = 0.0
        self._prev_scorer_x = None
        self._prev_scorer_y = None

    def reset(self, scorer_x, scorer_y):
        dx = GOAL_X - scorer_x
        dy = GOAL_Y - scorer_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            self.x = scorer_x
            self.y = scorer_y
        else:
            ux, uy = dx / dist, dy / dist
            self.x = scorer_x + 0.6 * ux
            self.y = scorer_y + 0.6 * uy
        self.x = max(COURT_X_MIN, min(COURT_X_MAX, self.x))
        self.y = max(COURT_Y_MIN, min(COURT_Y_MAX, self.y))
        self.yaw = 0.0
        self._prev_scorer_x = None
        self._prev_scorer_y = None
        self.scorer_vx = 0.0
        self.scorer_vy = 0.0

    def _get_blocking_point(self, scorer_x, scorer_y):
        dx = GOAL_X - scorer_x
        dy = GOAL_Y - scorer_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return scorer_x, scorer_y
        ux, uy = dx / dist, dy / dist
        tx = scorer_x + 0.6 * ux
        ty = scorer_y + 0.6 * uy
        tx = max(COURT_X_MIN, min(COURT_X_MAX, tx))
        ty = max(COURT_Y_MIN, min(COURT_Y_MAX, ty))
        return tx, ty

    def step(self, scorer_x, scorer_y):
        if self._prev_scorer_x is not None:
            self.scorer_vx = (scorer_x - self._prev_scorer_x) / STEP_DT
            self.scorer_vy = (scorer_y - self._prev_scorer_y) / STEP_DT
        self._prev_scorer_x = scorer_x
        self._prev_scorer_y = scorer_y

        tx, ty = self._get_blocking_point(scorer_x, scorer_y)
        dist_to_block = math.sqrt((self.x - tx)**2 + (self.y - ty)**2)
        dist_scorer_to_goal = math.sqrt((GOAL_X - scorer_x)**2 + (GOAL_Y - scorer_y)**2)

        obs = np.array([
            self.x, self.y, self.yaw,
            scorer_x, scorer_y,
            self.scorer_vx, self.scorer_vy,
            GOAL_X, GOAL_Y,
            dist_to_block,
            dist_scorer_to_goal,
        ], dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        linear_vel = float(action[0])
        angular_vel = float(action[1])

        self.yaw += angular_vel * STEP_DT
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.x += linear_vel * math.cos(self.yaw) * STEP_DT
        self.y += linear_vel * math.sin(self.yaw) * STEP_DT
        self.x = max(COURT_X_MIN, min(COURT_X_MAX, self.x))
        self.y = max(COURT_Y_MIN, min(COURT_Y_MAX, self.y))


class ScorerEnvSimple(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, defender_model_path: str = "defender_ppo_hpc_final"):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-MAX_LINEAR, -MAX_ANGULAR], dtype=np.float32),
            high=np.array([MAX_LINEAR, MAX_ANGULAR], dtype=np.float32),
            dtype=np.float32
        )

        # 10 observations - scorer can see defender position
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -10, -10,
                          -10, -10, 0, 0, -math.pi], dtype=np.float32),
            high=np.array([10, 10, math.pi, 10, 10,
                           10, 10, 20, 20, math.pi], dtype=np.float32),
            dtype=np.float32
        )

        self.defender = DefenderSim(defender_model_path)
        self.robot_x = 1.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.current_step = 0
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0

    def _get_obs(self):
        dist_to_paint = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )
        dist_to_defender = math.sqrt(
            (self.defender.x - self.robot_x) ** 2 +
            (self.defender.y - self.robot_y) ** 2
        )
        angle_to_goal = math.atan2(GOAL_Y - self.robot_y, GOAL_X - self.robot_x)
        heading_error = angle_to_goal - self.robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        return np.array([
            self.robot_x, self.robot_y, self.robot_yaw,
            self.defender.x, self.defender.y,
            GOAL_X, GOAL_Y,
            dist_to_paint,
            dist_to_defender,
            heading_error,
        ], dtype=np.float32)

    def _compute_reward(self):
        dist_to_paint = math.sqrt(
            (GOAL_X - self.robot_x) ** 2 + (GOAL_Y - self.robot_y) ** 2
        )
        dist_to_defender = math.sqrt(
            (self.defender.x - self.robot_x) ** 2 +
            (self.defender.y - self.robot_y) ** 2
        )

        # Primary: get to the paint
        paint_reward = 5.0 * math.exp(-1.5 * dist_to_paint)

        # Big bonus for reaching paint
        paint_bonus = 30.0 if dist_to_paint <= PAINT_RADIUS else 0.0

        # Facing reward
        angle_to_goal = math.atan2(GOAL_Y - self.robot_y, GOAL_X - self.robot_x)
        heading_error = angle_to_goal - self.robot_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        facing_reward = 0.5 * math.cos(heading_error)

        # Collision penalty with defender
        collision_penalty = -10.0 if dist_to_defender < 0.4 else 0.0

        # Out of bounds penalty
        out_of_bounds = (
            self.robot_x < COURT_X_MIN or self.robot_x > COURT_X_MAX or
            self.robot_y < COURT_Y_MIN or self.robot_y > COURT_Y_MAX
        )
        bounds_penalty = -10.0 if out_of_bounds else 0.0

        # Time penalty
        time_penalty = -0.05

        return (paint_reward + paint_bonus + facing_reward +
                collision_penalty + bounds_penalty + time_penalty)

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

    def step(self, action):
        self.current_step += 1
        self.last_linear_vel = float(action[0])
        self.last_angular_vel = float(action[1])

        self._apply_action(self.last_linear_vel, self.last_angular_vel)
        self.defender.step(self.robot_x, self.robot_y)

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

        sx, sy = SCORER_STARTS[np.random.randint(len(SCORER_STARTS))]
        self.robot_x = sx + np.random.uniform(-0.2, 0.2)
        self.robot_y = sy + np.random.uniform(-0.2, 0.2)
        self.robot_yaw = np.random.uniform(-math.pi, math.pi)

        self.defender.reset(self.robot_x, self.robot_y)

        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0

        return self._get_obs(), {}

    def close(self):
        pass
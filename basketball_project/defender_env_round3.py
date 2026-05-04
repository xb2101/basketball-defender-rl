#!/usr/bin/env python3
"""
defender_env_round3.py — Pure Python defender environment for Round 3 training.

Round 3: Defender retrains against a frozen trained scorer (PPO v7).
Replaces the scripted ScorerSim with a frozen PPO scorer model.

Key differences from defender_env_simple.py:
- ScorerSim replaced with ScorerPPO: loads a frozen scorer PPO model
  and uses it to generate scorer actions each step
- Scorer observation space is 10 dims (matches scorer_env_simple.py v7+)
- Everything else (reward function, defender obs space, spawn ranges) is
  identical to the v6 defender that achieved 30M steps

Observation space (11 dims, unchanged):
[robot_x, robot_y, robot_yaw,
 scorer_x, scorer_y,
 scorer_vx, scorer_vy,
 goal_x, goal_y,
 dist_to_block, dist_scorer_to_goal]
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Constants — must match Gazebo setup exactly
# ---------------------------------------------------------------------------
COURT_X_MIN, COURT_X_MAX = 0.0, 5.0
COURT_Y_MIN, COURT_Y_MAX = -4.0, 4.0
GOAL_X, GOAL_Y = 5.0, 0.0
PAINT_RADIUS = 0.8
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


class ScorerPPO:
    """Simulates the trained scorer using a frozen PPO model (v7)."""

    def __init__(self, model_path: str):
        print(f"Loading frozen scorer model from {model_path}...")
        self.model = PPO.load(model_path, device='cpu')
        self.x = 1.0
        self.y = 0.0
        self.yaw = 0.0
        self._prev_x = None
        self._prev_y = None
        self.vx = 0.0
        self.vy = 0.0
        # Defender position needed for scorer observation
        self.defender_x = 3.0
        self.defender_y = 0.0

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.yaw = np.random.uniform(-math.pi, math.pi)
        self._prev_x = None
        self._prev_y = None
        self.vx = 0.0
        self.vy = 0.0

    def update_defender_pos(self, defender_x, defender_y):
        """Called each step so scorer can see defender position."""
        self.defender_x = defender_x
        self.defender_y = defender_y

    def _get_obs(self):
        """Build scorer's 10-dim observation (matches scorer_env_simple.py)."""
        dist_to_paint = math.sqrt(
            (GOAL_X - self.x) ** 2 + (GOAL_Y - self.y) ** 2
        )
        dist_to_defender = math.sqrt(
            (self.defender_x - self.x) ** 2 +
            (self.defender_y - self.y) ** 2
        )
        angle_to_goal = math.atan2(GOAL_Y - self.y, GOAL_X - self.x)
        heading_error = angle_to_goal - self.yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        return np.array([
            self.x, self.y, self.yaw,
            self.defender_x, self.defender_y,
            GOAL_X, GOAL_Y,
            dist_to_paint,
            dist_to_defender,
            heading_error,
        ], dtype=np.float32)

    def step(self):
        """Step the scorer using its frozen PPO policy. Returns True if scored."""
        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        linear_vel = float(action[0])
        angular_vel = float(action[1])

        self.yaw += angular_vel * STEP_DT
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi

        prev_x, prev_y = self.x, self.y
        self.x += linear_vel * math.cos(self.yaw) * STEP_DT
        self.y += linear_vel * math.sin(self.yaw) * STEP_DT
        self.x = max(COURT_X_MIN, min(COURT_X_MAX, self.x))
        self.y = max(COURT_Y_MIN, min(COURT_Y_MAX, self.y))

        self.vx = (self.x - prev_x) / STEP_DT
        self.vy = (self.y - prev_y) / STEP_DT

        dist_to_goal = math.sqrt((GOAL_X - self.x) ** 2 + (GOAL_Y - self.y) ** 2)
        return dist_to_goal <= PAINT_RADIUS


class DefenderEnvRound3(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scorer_model_path: str = "scorer_ppo_hpc_v7_final"):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-MAX_LINEAR, -MAX_ANGULAR], dtype=np.float32),
            high=np.array([MAX_LINEAR, MAX_ANGULAR], dtype=np.float32),
            dtype=np.float32
        )

        # 11 dims — identical to defender_env_simple.py
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -math.pi, -10, -10,
                          -2, -2, -10, -10, 0, 0], dtype=np.float32),
            high=np.array([10, 10, math.pi, 10, 10,
                           2, 2, 10, 10, 20, 20], dtype=np.float32),
            dtype=np.float32
        )

        self.scorer = ScorerPPO(scorer_model_path)
        self.robot_x = 2.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
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
            self.scorer.vx, self.scorer.vy,
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

        # Goal penalty — big penalty if scorer reaches paint
        goal_penalty = -15.0 if self._scorer_reached_paint() else 0.0

        # Time penalty
        time_penalty = -0.05

        # Smoothness penalty
        smoothness_penalty = -1.5 * abs(self.last_angular_vel)

        # Velocity penalty when close to blocking point
        velocity_penalty = -1.0 * abs(self.last_linear_vel) if dist_to_block < 0.5 else 0.0

        return (blocking_reward + facing_reward +
                interception_reward + collision_penalty +
                bounds_penalty + goal_penalty + time_penalty +
                smoothness_penalty + velocity_penalty)

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

        # Give scorer current defender position before it steps
        self.scorer.update_defender_pos(self.robot_x, self.robot_y)
        scored = self.scorer.step()

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

        sx, sy = SCORER_STARTS[np.random.randint(len(SCORER_STARTS))]
        self.scorer.reset(sx, sy)

        return self._get_obs(), {}

    def close(self):
        pass

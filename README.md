# Teaching Robots Basketball: A Multi-Agent RL Approach

A reinforcement learning project where two TurtleBot3 robots learn to compete on a basketball court — one tries to score by reaching the paint, the other tries to defend by blocking it.

Built with ROS2, Gazebo, Stable Baselines3 (PPO), and trained on NYU HPC.

---

## Project Overview

| Agent | Goal |
|-------|------|
| **Defender** | Track the scorer and position itself between the scorer and the goal |
| **Scorer** | Navigate from the 3-point arc to the paint (goal area) |

The robots are trained using **Proximal Policy Optimization (PPO)** with an **alternating frozen training** approach — one agent trains while the other stays frozen, then they swap. This allows each agent to adapt to a progressively stronger opponent.

---

## Training Approach: 3 Rounds

### Round 1 — Defender Training
- Defender trains against a **scripted scorer** that moves to random court positions before driving to the goal
- Trained for **30M steps** on NYU HPC
- Final model: `defender_hpc_v6_final`

### Round 2 — Scorer Training
- Scorer trains **alone** with no defender, learning basic court navigation
- Trained for **10M steps** on NYU HPC
- Final model: `scorer_ppo_hpc_v6_final`

### Round 3 — Frozen Alternating Training
- **Step 1**: Scorer trains against frozen Defender v6 → `scorer_ppo_hpc_v7_final`
- **Step 2**: Defender retrains against frozen Scorer v7 → `defender_ppo_r3_final`

---

## Repository Structure

```
basketball_project/
├── defender_rl_env.py        # Gazebo-based defender environment (ROS2)
├── defender_env_simple.py    # Pure Python defender environment (HPC training)
├── defender_env_round3.py    # Pure Python defender env for Round 3 (vs frozen scorer)
├── scorer_env_simple.py      # Pure Python scorer environment (HPC training)
├── scorer_controller.py      # Scripted scorer controller (Round 1)
├── run_model.py              # Run trained defender in Gazebo
├── run_scorer.py             # Run trained scorer in Gazebo
├── train_hpc.py              # Defender training script (NYU HPC)
├── train_scorer_hpc.py       # Scorer training script (NYU HPC)
├── train_defender_round3.py  # Round 3 defender training script (NYU HPC)
├── train_defender_ppo.py     # Early local training script (pre-HPC)
launch/
├── sim_rl.launch.py          # Main Gazebo launch file
worlds/
├── basketball_court.world    # Basketball court world
train_defender.slurm          # SLURM job script for defender training
train_scorer.slurm            # SLURM job script for scorer training
```

---

## Requirements

- Ubuntu 20.04
- ROS2 Foxy
- Gazebo 11
- TurtleBot3 packages
- Python 3.8+
- `stable-baselines3`
- `gymnasium`
- `numpy`

Install Python dependencies:
```bash
pip install stable-baselines3 gymnasium numpy
```

---

## Running the Demo (Round 3)

> **Note**: Run `source ~/turtlebot3_ws/install/setup.bash` in every new terminal before running any command.

**Terminal 1** — Launch Gazebo:
```bash
source ~/turtlebot3_ws/install/setup.bash
ros2 launch basketball_project sim_rl.launch.py
```

**Terminal 2** — Run the defender:
```bash
source ~/turtlebot3_ws/install/setup.bash
cd ~/turtlebot3_ws/src/basketball_project/basketball_project/
python3 run_model.py
```

**Terminal 3** — Run the scorer:
```bash
source ~/turtlebot3_ws/install/setup.bash
cd ~/turtlebot3_ws/src/basketball_project/basketball_project/
python3 run_scorer.py --checkpoint scorer_hpc_v7_final --obs 10
```

---

## Running Round 1 Demo (Defender vs Scripted Scorer)

In `sim_rl.launch.py`, uncomment `spawn_scorer` and `start_scorer_controller`, and comment out `spawn_scorer_robot`.

In `defender_rl_env.py`, change:
```python
self.scorer_name = 'scorer_robot'
```
to:
```python
self.scorer_name = 'scorer'
```

Then rebuild and launch:
```bash
cd ~/turtlebot3_ws
colcon build --packages-select basketball_project
source install/setup.bash
ros2 launch basketball_project sim_rl.launch.py
```

**Terminal 2** — Run the defender:
```bash
python3 run_model.py
```

---

## Running Round 2 Demo (Scorer Only)

In `sim_rl.launch.py`, comment out `spawn_defender` and `start_scorer_controller`. Keep `spawn_scorer_robot` uncommented.

Then rebuild and launch Gazebo, then run:
```bash
python3 run_scorer.py --checkpoint scorer_phase1_final --obs 7
```

---

## HPC Training (NYU Torch Cluster)


All training files are in `/scratch/xb2101/basketball_project/`.

**Train defender (Round 1):**
```bash
sbatch train_defender.slurm
```

**Train scorer (Round 2/3):**
```bash
sbatch train_scorer.slurm
```

**Monitor jobs:**
```bash
squeue --me
tail -f logs/defender_<jobid>.out
```

---

## Git Tags

| Tag | Description |
|-----|-------------|
| `round1-defender` | Defender training setup against scripted scorer |
| `round2-scorer` | Scorer only training setup |
| `round3-frozen` | Full frozen alternating training setup |

---

## Court Specifications

| Parameter | Value |
|-----------|-------|
| Court bounds | x: 0–5, y: −4 to 4 |
| Goal position | (5.0, 0.0) |
| Paint radius | 1.0m |
| Blocking point | 0.6m ahead of scorer toward goal |

---

## Challenges

- **No real physics in training**: Robots trained in pure Python simulation pass through each other. Collision penalties are reward signals, not physical barriers.
- **Sim-to-real gap**: Behaviors trained in Python sim may differ from Gazebo due to missing physics.
- **Scorer ignores defender**: The paint reward consistently outweighs the collision penalty, so the scorer learned to go straight regardless of the defender's position.

---

## Authors

Xavier Beltran — NYU Tandon School of Engineering, Spring 2026  

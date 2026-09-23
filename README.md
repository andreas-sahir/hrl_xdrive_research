# Hierarchical Reinforcement Learning for Dynamic Navigation and Kinematic Drift Compensation in X-Drive Robot

This repository contains the official ROS 2 Humble and Gazebo Classic 11 research implementation for training and evaluating a Hierarchical Reinforcement Learning (HRL) architecture on an omnidirectional X-drive mobile robot.

The project addresses the challenge of non-linear kinematic drift, anisotropic friction, and dynamic obstacle avoidance. By replacing traditional model-based controllers with a unified multivariable (MIMO) Deep Reinforcement Learning framework, this codebase decouples cognitive pathfinding from high-frequency kinematic execution via a Semi-Markov Decision Process (SMDP).

## Project Vision and Research Questions

The X-drive mobile platform employs four omni-wheels positioned diagonally at $45^\circ$ angles on a $60\text{ cm} \times 60\text{ cm}$ chassis. While providing 3-DoF holonomic maneuverability in $SE(2)$, real-world interactions between the driven hub and passive rollers create severe anisotropic friction ($\mu_1 = 0.20$ longitudinal grip, $\mu_2 = 0.01$ lateral roller slip).

Classical decoupled Single-Input Single-Output (SISO) PID controllers fight this natural holonomic drift, generating destructive actuator interference that leads to terminal stalling in high-friction maneuvers. This research codebase proves that a unified MIMO DRL controller autonomously discovers momentum-conserving strategies that preserve kinetic energy and neutralize kinematic drift.

The codebase is engineered to answer two fundamental research questions:

- Kinematic Drift Compensation: How can a unified, multivariable DRL control strategy effectively compensate for non-linear kinematic drift in X-drive robots, overcoming the terminal stalling and actuator fighting inherent to classical decoupled (SISO) controllers?
- Hierarchical Dynamic Navigation: How can an HRL architecture, modeled as a Semi-Markov Decision Process (SMDP), be structured to simultaneously integrate reactive dynamic obstacle avoidance with low-level drift compensation to ensure safe navigation in complex, wall-bound topologies?

## Core Architectural Principles

The repository is structured around a three-tiered control hierarchy:

### 1. Macroscopic Global Planning (A* Search & Sparsification)

Operates offline on an inflated configuration space (C<sub>space</sub>) constructed via Minkowski sums, where C<sub>obs</sub> = O ⊕ (-A).

Applies a custom trajectory sparsification filter (1.5 m to 2.0 m spacing) to provide the strategic agent with a sliding spatial lookahead matrix of upcoming waypoints (W<sub>1</sub>, W<sub>2</sub>, W<sub>3</sub>), resolving myopic execution traps.

### 2. High-Level Strategic Policy (SMDP Soft Actor-Critic / SAC)

Functions as the cognitive navigator operating at lower execution frequencies (1 Hz to 5 Hz).

Addresses Partial Observability (POMDP) by employing temporal state representation through frame stacking (N = 4), allowing the policy to infer the velocity and momentum of dynamic threats from raw LiDAR scans.

Outputs a 3D continuous macro-action: localized Cartesian displacement offsets (dx, dy) and a dynamic speed scaling factor (v<sub>scale</sub> in [0.1, 1.0]).

### 3. Low-Level Tactical Controller (Discrete PPO)

Functions as a high-frequency ($50\text{ Hz}$) smart actuator with frozen policy weights during high-level training to maintain environment stationarity.

Employs an 11-action discrete space (terminal braking, cardinal translations, $\sin(45^\circ)$-normalized diagonal translations, and pure rotations) mapped to global chassis velocities ($v_x, v_y, \omega_z$) via Inverse Kinematics.

Serves as an architectural regularizer that prevents destabilizing motor jitter and achieves terminal braking precision within a $0.08\text{ m}$ acquisition radius.

## Key Contributions and Findings

- Momentum Conservation vs. Terminal Stalling: Demonstrates that while classical PID controllers achieve a low success rate ($25.0\%$) due to actuator cancellation in high friction, discrete PPO achieves an $87.5\%$ success rate by permitting controlled lateral sideslip to overcome static friction.
- SMDP Temporal Abstraction: Decouples high-level strategic reasoning from low-level execution, achieving a $100.0\%$ navigation success rate in open environments populated by sinusoidal dynamic obstacles (Simple Harmonic Motion).
- Sub-Centimeter Drift Compensation: In complex wall-bound maze topologies (Stage 4), the frozen low-level controller maintains an average Cross-Track Error (CTE) of $0.0077\text{ m}$ (less than $1\text{ cm}$ of lateral deviation), proving complete neutralization of physical slipping.
- Curriculum Learning Pipeline: Features a sequential 4-stage evolutionary training pipeline (Empty Arena $\rightarrow$ Static Pillars $\rightarrow$ Dynamic Threats $\rightarrow$ Wall-Bound Maze) that prevents early policy collapse and builds robust spatial generalization.


## Repository layout

```text
hrl_xdrive_research/
├── config/                      # ROS 2 YAML settings for controllers, EKF, mapping, and launch config
│   ├── controllers.yaml
│   ├── ekf.yaml
│   ├── slam.yaml
│   └── ...
├── description/                 # Robot description files and URDF assembly assets
│   ├── urdf/
│   ├── meshes/
│   └── ...
├── launch/                      # ROS 2 launch files for simulation, display, and odometry flows
│   ├── display.launch.py
│   ├── headless.launch.py
│   ├── hrl.launch.py
│   ├── odom.launch.py
│   └── simulation.launch.py
├── maps/                        # Occupancy-grid maps and related map metadata
│   ├── stage4_map.pgm
│   ├── stage4_map.yaml
│   └── ...
├── navigations/                 # Navigation and error-analysis utilities
│   └── error_calculation.py
├── planning/                    # Global planner and map-generation logic
│   ├── astar_planner.py
│   └── stage4_map.py
├── rl/                          # Reinforcement learning pipeline
│   ├── environments/
│   │   ├── continous_tracker_env.py
│   │   ├── ddqn_env.py
│   │   ├── hrl_env.py
│   │   ├── hrl_env123.py
│   │   ├── omni_rl_env.py
│   │   └── velocity_tracking_env.py
│   ├── evaluation/
│   │   ├── evaluate_dqn.py
│   │   ├── evaluate_goal.py
│   │   ├── evaluate_hrl.py
│   │   ├── evaluate_hrl123.py
│   │   ├── evaluate_ll.py
│   │   └── evaluate_rl.py
│   └── training/
│       ├── ddqn_train.py
│       ├── train_hl_sac.py
│       ├── train_ppo.py
│       └── train_sac_tracker.py
├── robot_control/               # Omni-drive kinematics, control, and odometry
│   ├── odometry_publisher.py
│   ├── omni_controller.py
│   └── omni_controller_nopid.py
├── rviz/                        # RViz visualization configuration
├── simulation_tools/            # Dynamic obstacle helpers used in simulation
│   └── obstacles/
├── tests/                       # Targeted validation and motion tests
│   └── test_nopid.py
├── worlds/                      # Gazebo world definitions
├── CMakeLists.txt
├── LICENSE
├── package.xml
├── README.md
└── .vscode/                     # Local editor settings for ROS venv compatibility
```

## System components

### 1. Global planning layer
The repository includes a custom A* planning pipeline used to build a collision-free route through known occupied space. The planner operates over a discretized occupancy grid and produces a sparse waypoint path that is more suitable for high-level control than a dense per-cell trajectory.

Relevant files:
- [planning/astar_planner.py](planning/astar_planner.py)
- [planning/stage4_map.py](planning/stage4_map.py)

### 2. High-level strategic controller
The high-level policy uses observations such as current pose, waypoint metrics, obstacle proximity, and lidar features to decide local target motions. This layer makes a lower-frequency strategic decision about which direction or local goal region to pursue.

Relevant files:
- [rl/environments/hrl_env.py](rl/environments/hrl_env.py)
- [rl/environments/hrl_env123.py](rl/environments/hrl_env123.py)
- [rl/training/train_hl_sac.py](rl/training/train_hl_sac.py)

### 3. Low-level reactive controller
The low-level policy learns direct commands in the robot's planar motion space. It is trained to exploit the robot's omni-directional degrees of freedom while compensating for anisotropic friction and sideways drift.

Relevant files:
- [rl/environments/omni_rl_env.py](rl/environments/omni_rl_env.py)
- [rl/environments/continous_tracker_env.py](rl/environments/continous_tracker_env.py)
- [rl/training/train_ppo.py](rl/training/train_ppo.py)
- [rl/evaluation/evaluate_goal.py](rl/evaluation/evaluate_goal.py)

### 4. Kinematics and robot interface
This layer converts desired body-frame velocity commands into wheel angular velocities and publishes robot odometry and transforms.

Relevant files:
- [robot_control/omni_controller.py](robot_control/omni_controller.py)
- [robot_control/omni_controller_nopid.py](robot_control/omni_controller_nopid.py)
- [robot_control/odometry_publisher.py](robot_control/odometry_publisher.py)

### 5. Navigation and monitoring
The repository includes performance monitoring and error decomposition to assess controller quality, navigation accuracy, and goal-reaching behavior during training and evaluation.

Relevant files:
- [navigations/error_calculation.py](navigations/error_calculation.py)
- [rl/evaluation/evaluate_hrl.py](rl/evaluation/evaluate_hrl.py)

## Simulation and ROS 2 workflow

The project is designed for ROS 2 Humble and Gazebo. The launch files are organized around testing and training workflows:

- `simulation.launch.py`: standard Gazebo simulation with robot state publication and controller setup
- `headless.launch.py`: headless simulation for faster RL training and automated runs
- `hrl.launch.py`: combined HRL stack launch for hierarchical navigation experiments
- `odom.launch.py`: odometry-related launch flow
- `display.launch.py`: GUI visual debugging and display launch

## Training modes supported

This repo includes training code for several policy types:

- PPO for low-level control and drift compensation
- SAC for high-level strategic policy learning
- DDQN for discrete decision learning experiments
- custom tracking and continuous-action environments for motion optimization

## Baselines and evaluation

The project includes classical baselines as comparison points:

- PID-controlled omnidirectional motion
- open-loop or no-PID motion tests
- direct tracking and waypoint-following evaluations
- collision and crash tests

Evaluation scripts are stored under:
- [rl/evaluation](rl/evaluation)
- [tests](tests)

## Requirements

### Operating system
- Ubuntu 22.04 LTS

### ROS 2
- ROS 2 Humble
- Gazebo and ROS control packages
- robot_localization, robot_state_publisher, controller_manager

### Python
- Python 3.10
- numpy, scipy, gymnasium, stable-baselines3
- PyTorch
- pyyaml
- Python dependencies for this project are also listed in [requirements.txt](requirements.txt).

## Installation

From a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
# clone repo here
# e.g. git clone <repo-url> my_robot
```

Then install dependencies:

```bash
sudo apt update
sudo apt install -y \
 ros-humble-gazebo-ros-pkgs \
 ros-humble-robot-state-publisher \
 ros-humble-controller-manager \
 ros-humble-ros2-control \
 ros-humble-ros2-controllers \
 ros-humble-robot-localization
```

Install Python tools in the active environment:

```bash
source /home/andreas/ros_venv/bin/activate
python -m pip install numpy scipy gymnasium stable-baselines3 torch pyyaml
```

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot
source install/setup.bash
```

## Typical usage

### Launch simulation

```bash
ros2 launch my_robot headless.launch.py
```

or

```bash
ros2 launch my_robot simulation.launch.py
```

### Train low-level controller

```bash
source /home/andreas/ros_venv/bin/activate
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
python rl/training/train_ppo.py
```

### Train high-level policy

```bash
python rl/training/train_hl_sac.py
```

### Run evaluation

```bash
python rl/evaluation/evaluate_goal.py
```

## Notes

- The repository is organized around the active ROS 2 workspace layout and reflects the current code structure rather than older script-based packaging.
- The project is research-oriented; several training and evaluation paths are intended for custom setups and may require path or environment adjustments depending on the local workspace layout.
- The package name is `my_robot` in the ROS 2 package metadata even though the repository root is named `hrl_xdrive_research`.

## Citation / source notice

This repository implements and extends the research work presented in:

A. S. Aryanto, A. Ataka, I. Ardiyanto and M. I. Zulfa, "Compensating for Kinematic Drift in X-Drive Mobile Robot Using Model-Free Reinforcement Learning," *2026 23rd International Conference on Electrical Engineering/Electronics, Computer, Telecommunications and Information Technology (ECTI-CON)*, Chonburi, Thailand, 2026, pp. 49-54, doi: 10.1109/ECTI-CON68836.2026.11608427.

Please cite this work when using or adapting the methods, environments, or results in this repository.

## Summary

This repository provides a complete research pipeline for learning and evaluating autonomous motion policies for an X-drive robot in ROS 2 and Gazebo. It combines simulation, planning, control, RL training, and evaluation into one workflow focused on omnidirectional navigation, drift compensation, and robust motion execution in constrained environments.
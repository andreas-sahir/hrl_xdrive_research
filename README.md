# Hierarchical Reinforcement Learning for Omni-Directional Navigation and Drift Compensation

This repository contains a ROS 2 + Gazebo research codebase for training and evaluating hierarchical reinforcement learning policies on an omnidirectional X-drive robot. The project focuses on autonomous navigation in dynamic and constrained environments, with particular emphasis on reducing kinematic drift, cross-coupled wheel effects, and control instability caused by anisotropic friction and actuator interactions.

The work is centered around a hierarchical control architecture that combines:

- a global planning layer for route generation,
- a high-level strategic policy for macro-level decision making,
- a low-level reactive controller for precise local execution.

The code supports both simulation-based experiments and training/evaluation workflows for PPO, SAC, DDQN, and classical baselines.

## Project vision

The robot platform in this repository is modeled as a 4-wheel omnidirectional X-drive system. Unlike standard differential-drive robots, this platform can move in multiple directions and rotate simultaneously, but it also exhibits complex nonlinear effects when executing lateral and diagonal maneuvers. This repository explores how reinforcement learning can learn compensation strategies that outperform classical decoupled PID control in trajectory tracking, obstacle avoidance, and motion stability.

The system is designed to answer the core research question:

- Can a hierarchical reinforcement learning policy learn to navigate constrained spaces while compensating for the robot's nonideal kinematics and friction-induced drift?

## Main ideas

- Hierarchical RL with macro and micro control loops
- Semi-Markov planning style behavior for goal-directed motion
- Dynamic obstacle handling through map and lidar-based observations
- Low-level policy trained to learn motion corrections for drift and slip
- High-level policy trained to choose long-horizon navigation behaviors
- Classical baselines included for comparison against reinforcement learning controllers

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

### Recommended environment
This project is intended to run inside the repository's ROS venv, for example:

```bash
source /home/andreas/ros_venv/bin/activate
source /opt/ros/humble/setup.bash
```

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
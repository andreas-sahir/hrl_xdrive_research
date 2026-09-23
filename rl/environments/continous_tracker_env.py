#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time
import csv
import os
from datetime import datetime
from std_srvs.srv import Empty
from robot_localization.srv import SetPose

class OmniRobotTrackerEnv(Node, gym.Env):
    def __init__(self, reward_type='full'):
        Node.__init__(self, 'omni_rl_continuous_controller')
        gym.Env.__init__(self)
        
        # --- CONFIGURATION  ---
        self.max_episode_steps = 500
        self.speed_limit = 1.0       # MATCHES PID BASELINE
        self.success_thresh = 0.08   # BEATS PID PRECISION
        # ------------------------
        
        self.current_step = 0
        self.has_received_first_odom = False
        self.prev_dist_to_goal = None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"sac_v2_pid_behavior_{timestamp}.csv"
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Step', 'Dist', 'Rew', 'R_Prog', 'R_Align', 'R_Succ', 'R_Drift'])
        self.get_logger().info(f"Logging V2 SAC data to: {self.csv_filename}")
        
        # Continuous Action Space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([3.0, 0.0])
        
        self.navigation_goals = [
            [3.0, 0.0], [-3.0, 0.0], [0.0, 3.0], [0.0, -3.0],
            [2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0],
            [4.0, 0.0], [-4.0, 0.0]
        ]
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.odom_sub = self.create_subscription(Odometry, '/ground_truth/pose', self.odom_callback, qos)
        self.reset_sim_client = self.create_client(Empty, '/reset_simulation')
        self.reset_ekf_client = self.create_client(SetPose, '/set_pose')

    def odom_callback(self, msg):
        if not self.has_received_first_odom:
            self.get_logger().info(f"[SAC] ODOMETRY CONNECTED! Pose: {msg.pose.pose.position.x:.2f}")
            self.has_received_first_odom = True
        self.current_velocity[0] = msg.twist.twist.linear.x
        self.current_velocity[1] = msg.twist.twist.linear.y
        self.current_velocity[2] = msg.twist.twist.angular.z
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_pose[2] = math.atan2(siny_cosp, cosy_cosp)

    def _get_observation(self):
        dx = self.goal_position[0] - self.current_pose[0]
        dy = self.goal_position[1] - self.current_pose[1]
        dist = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx)
        heading_err = angle_to_goal - self.current_pose[2]
        while heading_err > math.pi: heading_err -= 2 * math.pi
        while heading_err < -math.pi: heading_err += 2 * math.pi
        return np.array([
            dist, math.cos(heading_err), math.sin(heading_err),
            self.current_velocity[0], self.current_velocity[1], self.current_velocity[2]
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Robust Action Handling
        if isinstance(action, list): act = np.array(action)
        elif isinstance(action, np.ndarray): act = action.flatten() if action.ndim > 1 else action
        else: act = np.array([float(action), 0.0, 0.0])
        if act.size == 1: act = np.array([float(act[0]), 0.0, 0.0])
        if len(act) < 3: act = np.pad(act, (0, 3-len(act)))

        # Speed Scaling
        vx = float(act[0]) * self.speed_limit
        vy = float(act[1]) * self.speed_limit
        wz = float(act[2]) * self.speed_limit
        
        self._publish_velocity(vx, vy, wz)
        time.sleep(0.05) 
        rclpy.spin_once(self, timeout_sec=0.01)
        
        # --- REWARD FUNCTION  ---
        obs = self._get_observation()
        dist = obs[0]
        if self.prev_dist_to_goal is None: self.prev_dist_to_goal = dist
        
        # 1. Scaled Progress 
        r_progress = (self.prev_dist_to_goal - dist) * 2.0
        self.prev_dist_to_goal = dist
        
        # 2. Scaled Success
        success = dist < self.success_thresh
        r_success = 10.0 if success else 0.0
        
        # 3. PID Alignment Reward 
        dx_g = self.goal_position[0] - self.current_pose[0]
        dy_g = self.goal_position[1] - self.current_pose[1]
        goal_heading = math.atan2(dy_g, dx_g)
        
        vel_angle = math.atan2(self.current_velocity[1], self.current_velocity[0])
        vel_mag = math.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)
        
        # Reward heavily (+0.5) if velocity vector aligns with goal vector
        # This guides the continuous agent toward the correct heading
        alignment_score = math.cos(vel_angle - goal_heading)
        r_alignment = 0.5 * alignment_score if vel_mag > 0.1 else 0.0
        
        # 4. Scaled Penalties
        v_cross = vel_mag * math.sin(vel_angle - goal_heading)
        r_drift = -0.05 * abs(v_cross)     # Reduced from 0.1
        r_stable = -0.5 * abs(self.current_velocity[2]) # Reduced from 2.0
        r_energy = -0.005 * (abs(vx) + abs(vy))
        
        reward = r_progress + r_success + r_alignment + r_drift + r_stable + r_energy - 0.01
        
        terminated = success
        truncated = self.current_step >= self.max_episode_steps
        
        if self.current_step % 20 == 0:
            print(f"Step: {self.current_step:03d} | Act: [{vx:.1f}, {vy:.1f}] | Dist: {dist:.4f} | Rew: {reward:.4f} | Align: {r_alignment:.4f}", flush=True)

        self.csv_writer.writerow([self.current_step, f"{dist:.4f}", f"{reward:.4f}", f"{r_progress:.4f}", f"{r_alignment:.4f}", f"{r_success:.4f}", f"{r_drift:.4f}"])
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        req = Empty.Request(); future = self.reset_sim_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        req_ekf = SetPose.Request(); req_ekf.pose.header.frame_id = 'odom'; req_ekf.pose.pose.pose.orientation.w = 1.0
        future_ekf = self.reset_ekf_client.call_async(req_ekf)
        rclpy.spin_until_future_complete(self, future_ekf, timeout_sec=1.0)
        
        self.goal_position = np.array(self.navigation_goals[np.random.randint(len(self.navigation_goals))])
        
        self._publish_velocity(0.0, 0.0, 0.0)
        time.sleep(0.5); rclpy.spin_once(self, timeout_sec=0.1)
        self.prev_dist_to_goal = None
        return self._get_observation(), {}

    def _publish_velocity(self, vx, vy, wz):
        msg = Twist(); msg.linear.x = float(vx); msg.linear.y = float(vy); msg.angular.z = float(wz)
        self.cmd_vel_pub.publish(msg)
    
    def close(self):
        self._publish_velocity(0.0, 0.0, 0.0)
        self.csv_file.close()
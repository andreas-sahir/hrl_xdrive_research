#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
import csv
from datetime import datetime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 
from rclpy.parameter import Parameter 

class HRLNavEnv(Node, gym.Env):
    def __init__(self, ll_model_path, ll_stats_path, stage='4', realtime_eval=False): 
        Node.__init__(self, 'hrl_navigator_env')
        gym.Env.__init__(self)
        
        self.stage = str(stage)

        self.realtime_eval = realtime_eval
        self.spin_timeout = 0.01 if realtime_eval else 0.0  
        
        # 1. ENABLE GAZEBO SIM TIME & TELEPORTATION
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # 2. LOAD FROZEN LOW-LEVEL AGENT
        self.ll_agent = PPO.load(ll_model_path)
        
        class DummyLL(gym.Env):
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            action_space = spaces.Discrete(11)
            
        self.ll_normalizer = VecNormalize.load(ll_stats_path, DummyVecEnv([lambda: DummyLL()]))
        self.ll_normalizer.training = False 
        self.ll_normalizer.norm_reward = False 
        
        self.get_logger().info("Frozen LL Agent and Normalizer Loaded Successfully.")
        
        # 3. SPACES
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        # 4. ENVIRONMENT VARIABLES
        self.current_step = 0
        self.max_steps = 250
        self.pose = np.zeros(3)
        self.vel = np.zeros(3)
        self.lidar_data = np.ones(24) * 5.0
        
        # CURRICULUM STAGE 1 & 2: Static Goal
        self.global_goal = np.array([4.0, 4.0]) 
        
        # Logging Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = open(f"hrl_hl_behavior_{timestamp}.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['MacroStep', 'TargetX', 'TargetY', 'DistToGlobal', 'Reward', 'Collision'])

        # ROS 2 Pubs & Subs
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.odom_sub = self.create_subscription(Odometry, '/ground_truth/pose', self.odom_cb, qos)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)

    def odom_cb(self, msg):
        self.pose[0] = msg.pose.pose.position.x
        self.pose[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.pose[2] = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        self.vel[0] = msg.twist.twist.linear.x
        self.vel[1] = msg.twist.twist.linear.y
        self.vel[2] = msg.twist.twist.angular.z

    def lidar_cb(self, msg):
        raw_ranges = np.array(msg.ranges)
        raw_ranges[np.isnan(raw_ranges)] = 5.0
        raw_ranges[np.isinf(raw_ranges)] = 5.0
        raw_ranges[raw_ranges < 0.36] = 5.0 # Ignore robot's own chassis
        splits = np.array_split(raw_ranges, 24)
        self.lidar_data = np.array([np.min(bucket) for bucket in splits])

    def step(self, macro_action):
        self.current_step += 1
        
        # Calculate local target relative to chassis
        local_tx = self.pose[0] + (macro_action[0] * math.cos(self.pose[2]) - macro_action[1] * math.sin(self.pose[2]))
        local_ty = self.pose[1] + (macro_action[0] * math.sin(self.pose[2]) + macro_action[1] * math.cos(self.pose[2]))
        
        inner_steps = 0
        collision = False
        reached_local = False
        
        # --- LOW LEVEL SMDP LOOP ---
        while inner_steps < 20 and not collision and not reached_local:
            # FAST SIM ADJUSTMENT: Use non-blocking spin (timeout=0.0) so we don't wait
            # in wall-clock time while Gazebo is running ahead at max speed.
            rclpy.spin_once(self, timeout_sec=self.spin_timeout)
            
            dx = local_tx - self.pose[0]
            dy = local_ty - self.pose[1]
            dist_ll = math.sqrt(dx**2 + dy**2)
            head_err_ll = math.atan2(dy, dx) - self.pose[2]
            head_err_ll = math.atan2(math.sin(head_err_ll), math.cos(head_err_ll))
            
            raw_ll_obs = np.array([dist_ll, math.cos(head_err_ll), math.sin(head_err_ll), self.vel[0], self.vel[1], self.vel[2]], dtype=np.float32)
            norm_ll_obs = self.ll_normalizer.normalize_obs(raw_ll_obs)
            
            action, _ = self.ll_agent.predict(norm_ll_obs, deterministic=True)
            action_idx = int(action.item()) if hasattr(action, 'item') else int(action)
            
            msg = Twist()
            spd = 1.0; d_spd = 0.707
            if action_idx == 1: msg.linear.x = spd      
            elif action_idx == 2: msg.linear.x = -spd   
            elif action_idx == 3: msg.linear.y = spd    
            elif action_idx == 4: msg.linear.y = -spd   
            elif action_idx == 5: msg.linear.x = d_spd; msg.linear.y = d_spd   
            elif action_idx == 6: msg.linear.x = d_spd; msg.linear.y = -d_spd
            elif action_idx == 7: msg.linear.x = -d_spd; msg.linear.y = d_spd
            elif action_idx == 8: msg.linear.x = -d_spd; msg.linear.y = -d_spd
            elif action_idx == 9: msg.angular.z = spd   
            elif action_idx == 10: msg.angular.z = -spd 
            self.cmd_vel_pub.publish(msg)
            
            if dist_ll < 0.15: 
                reached_local = True
                
            # KILL SWITCH: Stop motors immediately on crash
            if np.min(self.lidar_data) < 0.45:
                collision = True
                self.cmd_vel_pub.publish(Twist())
                # FAST SIM ADJUSTMENT: Non-blocking spin to let the zero-velocity command publish
                rclpy.spin_once(self, timeout_sec=self.spin_timeout)
                break
            
            # FAST SIM ADJUSTMENT: Removed time.sleep(0.001) — with real_time_update_rate=0,
            # Gazebo runs as fast as the CPU allows. Wall-clock sleeps only slow training down.
            inner_steps += 1

        # --- EPISODE EVALUATION & REWARD (Corrected Math) ---
        dist_to_global = math.hypot(self.global_goal[0] - self.pose[0], self.global_goal[1] - self.pose[1])
        terminated = False
        truncated = False
        
        if collision:
            reward = -100.0
            terminated = True
            print(f"🛑 COLLISION! LiDAR Min: {np.min(self.lidar_data):.3f}m | Macro-Step: {self.current_step}")
        elif dist_to_global < 0.45:
            reward = 100.0
            terminated = True
            collision = False
            print(f"✅ GOAL REACHED! | Macro-Step: {self.current_step}")
        elif self.current_step >= self.max_steps: 
            reward = -1.0
            truncated = True
            print(f"⏱️ TIMEOUT! | Dist to Goal: {dist_to_global:.2f}m")
        else:
            reward = -1.0

        # --- RELATIVE OBSERVATION SPACE ---
        angle_to_global = math.atan2(self.global_goal[1] - self.pose[1], self.global_goal[0] - self.pose[0])
        hl_head_err = math.atan2(math.sin(angle_to_global - self.pose[2]), math.cos(angle_to_global - self.pose[2]))
        
        robot_state = np.array([dist_to_global, math.cos(hl_head_err), math.sin(hl_head_err), self.vel[0], self.vel[1], self.vel[2], 1.0])
        obs = np.concatenate((robot_state, self.lidar_data)).astype(np.float32)
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Stop physics
        self.cmd_vel_pub.publish(Twist())
        
        # Teleport protocol
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /gazebo/set_entity_state service...')
            
        req = SetEntityState.Request()
        req.state.name = 'my_robot'
        req.state.reference_frame = 'world' 
        req.state.pose.position.x = 0.0
        req.state.pose.position.y = 0.0
        req.state.pose.position.z = 0.05 
        req.state.pose.orientation.w = 1.0 
        
        # Call and wait for Gazebo to process
        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        # Clear fake crash data
        self.lidar_data = np.ones(24) * 5.0
        
        # FAST SIM ADJUSTMENT: Flush ROS buffers with non-blocking spins.
        # With real_time_update_rate=0, we don't want to burn wall-clock time waiting.
        for _ in range(15):
            rclpy.spin_once(self, timeout_sec=self.spin_timeout)

        # 🌟 CURRICULUM WAYPOINT GENERATOR 🌟
        if self.stage in ['1', '2', '3a']:
            # Random goal inside arena, not too close to spawn or walls
            while True:
                angle = np.random.uniform(-math.pi, math.pi)
                distance = np.random.uniform(2.0, 4.5)  # 2 m min so it's not trivial
                
                goal_x = self.pose[0] + (distance * math.cos(angle))
                goal_y = self.pose[1] + (distance * math.sin(angle))
                
                # Keep inside the inner safe bounds (walls are at ±5.5)
                if -4.5 < goal_x < 4.5 and -4.5 < goal_y < 4.5:
                    # Exclude spawn safe zone so goal isn't under the robot
                    if not (-1.0 < goal_x < 1.0 and -1.0 < goal_y < 1.0):
                        break
                        
            self.global_goal = np.array([goal_x, goal_y])
        
        # if self.stage in ['1', '2', '3a']:
        #         goal_x = 4.0 
        #         goal_y = 4.0
        #         self.global_goal = np.array([goal_x, goal_y])
            
        elif self.stage in ['3b', '4']:
            while True:
                angle = np.random.uniform(-math.pi, math.pi)
                distance = np.random.uniform(1.5, 3.0)
                
                goal_x = self.pose[0] + (distance * math.cos(angle))
                goal_y = self.pose[1] + (distance * math.sin(angle))
                
                if -4.5 < goal_x < 4.5 and -4.5 < goal_y < 4.5:
                    if not (-1.0 < goal_x < 1.0 and -1.0 < goal_y < 1.0):  # Fixed typo: was -3.0
                        break
                        
            self.global_goal = np.array([goal_x, goal_y])       
        
        # Build initial safe observation
        angle_to_global = math.atan2(self.global_goal[1] - self.pose[1], self.global_goal[0] - self.pose[0])
        hl_head_err = math.atan2(math.sin(angle_to_global - self.pose[2]), math.cos(angle_to_global - self.pose[2]))
        
        robot_state = np.array([math.hypot(self.global_goal[0], self.global_goal[1]), math.cos(hl_head_err), math.sin(hl_head_err), 0.0, 0.0, 0.0, 1.0])
        obs = np.concatenate((robot_state, self.lidar_data)).astype(np.float32)
        
        return obs, {}
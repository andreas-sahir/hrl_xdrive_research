import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import time

class VelocityTrackerNode(Node):
    """
    ROS 2 Node for Low-Level Control.
    - Publishes: /cmd_vel (Motor commands)
    - Subscribes: /odom (Actual robot velocity)
    """
    def __init__(self):
        super().__init__('velocity_tracker_node')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.odom_vx = 0.0
        self.odom_vy = 0.0
        self.odom_w = 0.0

    def odom_callback(self, msg):
        self.odom_vx = msg.twist.twist.linear.x
        self.odom_vy = msg.twist.twist.linear.y
        self.odom_w = msg.twist.twist.angular.z

class VelocityTrackerEnv(gym.Env):
    def __init__(self):
        super(VelocityTrackerEnv, self).__init__()
        
        # --- ROS INIT CHECK ---
        if not rclpy.ok():
            rclpy.init()
        self.node = VelocityTrackerNode()
        
        # --- ACTION SPACE (Discrete 11) ---
        # [cite_start]Matches your Table I in the draft [cite: 93]
        self.action_space = spaces.Discrete(11)
        
        # --- OBSERVATION SPACE ---
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32)
        
        # Targets
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_w = 0.0
        
        self.step_count = 0
        self.max_steps = 500
        self.change_target_interval = 100 

    def _generate_target_velocity(self):
        """
        --- THE FIX IS HERE ---
        Instead of random floats, we pick a specific 'Ideal Action' 
        and set the target to match it. This makes the task SOLVABLE.
        """
        # Pick a random discrete action index to mimic
        ideal_action = random.randint(0, 10)
        
        # Calculate what the velocity WOULD be for this action
        # This ensures the target is physically possible for the robot
        speed = 1.0
        diag_speed = 0.707 # Matches Table I in draft
        
        if ideal_action == 0:   self.target_vx = 0.0; self.target_vy = 0.0; self.target_w = 0.0
        elif ideal_action == 1: self.target_vx = speed; self.target_vy = 0.0; self.target_w = 0.0
        elif ideal_action == 2: self.target_vx = -speed; self.target_vy = 0.0; self.target_w = 0.0
        elif ideal_action == 3: self.target_vx = 0.0; self.target_vy = speed; self.target_w = 0.0
        elif ideal_action == 4: self.target_vx = 0.0; self.target_vy = -speed; self.target_w = 0.0
        
        # Diagonals (Important: Use 0.707 to match vector magnitude)
        elif ideal_action == 5: self.target_vx = diag_speed; self.target_vy = diag_speed; self.target_w = 0.0
        elif ideal_action == 6: self.target_vx = diag_speed; self.target_vy = -diag_speed; self.target_w = 0.0
        elif ideal_action == 7: self.target_vx = -diag_speed; self.target_vy = diag_speed; self.target_w = 0.0
        elif ideal_action == 8: self.target_vx = -diag_speed; self.target_vy = -diag_speed; self.target_w = 0.0
        
        # Rotations
        elif ideal_action == 9: self.target_vx = 0.0; self.target_vy = 0.0; self.target_w = 1.0
        elif ideal_action == 10: self.target_vx = 0.0; self.target_vy = 0.0; self.target_w = -1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        self.node.cmd_vel_pub.publish(Twist()) # Stop
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
        self._generate_target_velocity()
        return self.get_observation(), {}

    def step(self, action):
        self.step_count += 1
        
        if self.step_count % self.change_target_interval == 0:
            self._generate_target_velocity()

        self.take_action(action)
        time.sleep(0.05)
        rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self.get_observation()
        reward = self.calculate_reward(action)
        
        # --- LOGGING ---
        if self.step_count % 50 == 0:
            print(f"Step: {self.step_count:03d} | "
                  f"Tgt: [{self.target_vx:.2f}, {self.target_vy:.2f}] | "
                  f"Act: [{self.node.odom_vx:.2f}, {self.node.odom_vy:.2f}] | "
                  f"Rew: {reward:.4f}", flush=True)

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {}
        
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        return np.array([
            self.node.odom_vx, self.node.odom_vy, self.node.odom_w,
            self.target_vx, self.target_vy, self.target_w
        ], dtype=np.float32)

    def calculate_reward(self, action):
        v_curr = np.array([self.node.odom_vx, self.node.odom_vy])
        v_cmd  = np.array([self.target_vx, self.target_vy])
        
        # 1. TRACKING (Main)
        vel_error = np.linalg.norm(v_cmd - v_curr)
        r_track = -2.0 * vel_error 

        # 2. DRIFT
        p_drift = 0.0
        cmd_mag = np.linalg.norm(v_cmd)
        if cmd_mag > 0.05:
            v_cmd_norm = v_cmd / cmd_mag
            cross_prod = np.cross(v_curr, v_cmd_norm)
            p_drift = -1.5 * abs(cross_prod)

        # 3. STABILITY
        w_error = abs(self.target_w - self.node.odom_w)
        p_stable = -2.0 * w_error

        # 4. ENERGY
        p_energy = -0.01 if action != 0 else 0.0 # Action 0 is Stop

        return r_track + p_drift + p_stable + p_energy

    def take_action(self, action):
        msg = Twist()
        speed = 1.0
        diag_speed = 0.707 # 1.0 * sin(45)
        
        if action == 0: pass # Stop
        elif action == 1: msg.linear.x = speed      
        elif action == 2: msg.linear.x = -speed   
        elif action == 3: msg.linear.y = speed    
        elif action == 4: msg.linear.y = -speed   
        # Diagonals
        elif action == 5: msg.linear.x = diag_speed; msg.linear.y = diag_speed   
        elif action == 6: msg.linear.x = diag_speed; msg.linear.y = -diag_speed  
        elif action == 7: msg.linear.x = -diag_speed; msg.linear.y = diag_speed  
        elif action == 8: msg.linear.x = -diag_speed; msg.linear.y = -diag_speed 
        # Turns
        elif action == 9: msg.angular.z = speed   
        elif action == 10: msg.angular.z = -speed
        
        self.node.cmd_vel_pub.publish(msg)

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
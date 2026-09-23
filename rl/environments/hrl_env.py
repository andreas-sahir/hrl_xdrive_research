#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import csv
import os
from datetime import datetime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize 
from rclpy.parameter import Parameter 

try:
    from auto_explorer import AStarPlanner
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from auto_explorer import AStarPlanner


class HRLNavEnv(Node, gym.Env):
    def __init__(self, ll_model_path, ll_stats_path, stage='4', 
                 realtime_eval=False, map_path=None):
        Node.__init__(self, 'hrl_navigator_env')
        gym.Env.__init__(self)
        
        self.stage = str(stage)
        self.realtime_eval = realtime_eval
        self.spin_timeout = 0.01 if realtime_eval else 0.0  
        
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        self.ll_agent = PPO.load(ll_model_path)
        
        class DummyLL(gym.Env):
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
            action_space = spaces.Discrete(11)
            
        self.ll_normalizer = VecNormalize.load(ll_stats_path, DummyVecEnv([lambda: DummyLL()]))
        self.ll_normalizer.training = False 
        self.ll_normalizer.norm_reward = False 
        
        self.get_logger().info("Frozen LL Agent and Normalizer Loaded Successfully.")
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.1]),
            high=np.array([1.0, 1.0, 0.5]),
            dtype=np.float32
        )
        
        # ------------------------------------------------------------------
        # HIGH-LEVEL OBSERVATION SPACE
        # The state combines 9 waypoint metrics (WP1..WP3), the current linear
        # and angular body velocities, a terminal flag, and a 24-bin LiDAR scan.
        # This representation gives the macro-policy a richer understanding of the
        # path, local motion, and surrounding obstacle layout.
        # ------------------------------------------------------------------
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = 250
        self.pose = np.zeros(3)
        self.vel = np.zeros(3)
        self.lidar_data = np.ones(24) * 5.0
        
        self.global_goal = np.array([4.0, 4.0]) 
        self.prev_dist_to_global = None
        
        self.planner = None
        self.waypoints = []
        self.current_wp_idx = 0
        self.prev_remaining_path_length = None
        
        if map_path is not None and os.path.exists(map_path):
            self.planner = AStarPlanner(map_path, resolution=0.05, origin_x=-6.0, origin_y=-6.0)
            self.get_logger().info(f"A* Planner loaded from: {map_path}")
        else:
            self.get_logger().warn("No map_path provided; falling back to direct-goal navigation.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = open(f"hrl_hl_behavior_{timestamp}.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'MacroStep', 'TargetX', 'TargetY', 'DistToGoal', 
            'SpeedScale', 'Reward', 'Collision', 'WaypointIdx'
        ])

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
        raw_ranges[raw_ranges < 0.15] = 0.15  # Clamp close noise
        splits = np.array_split(raw_ranges, 24)
        self.lidar_data = np.array([np.min(bucket) for bucket in splits])

    def _get_lookahead_waypoints(self):
        """Returns the next 3 waypoints for trajectory lookahead."""
        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            return self.global_goal, self.global_goal, self.global_goal
            
        wp1 = self.waypoints[self.current_wp_idx]
        wp2 = self.waypoints[min(self.current_wp_idx + 1, len(self.waypoints) - 1)]
        wp3 = self.waypoints[min(self.current_wp_idx + 2, len(self.waypoints) - 1)]
        return wp1, wp2, wp3

    def _calc_target_metrics(self, target):
        """Calculates distance, cos(yaw), and sin(yaw) to a specific target."""
        dist = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        angle_to_target = math.atan2(target[1] - self.pose[1], target[0] - self.pose[0])
        head_err = math.atan2(math.sin(angle_to_target - self.pose[2]), math.cos(angle_to_target - self.pose[2]))
        return dist, math.cos(head_err), math.sin(head_err)

    def _compute_remaining_path_length(self):
        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            return math.hypot(self.global_goal[0] - self.pose[0], self.global_goal[1] - self.pose[1])
        target = self.waypoints[self.current_wp_idx]
        remaining = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        for i in range(self.current_wp_idx, len(self.waypoints) - 1):
            remaining += math.hypot(
                self.waypoints[i+1][0] - self.waypoints[i][0],
                self.waypoints[i+1][1] - self.waypoints[i][1]
            )
        return remaining

    def step(self, macro_action):
        self.current_step += 1
        
        dx = macro_action[0]
        dy = macro_action[1]
        speed_scale = float(np.clip(macro_action[2], 0.1, 1.0))

        min_obstacle = np.min(self.lidar_data)
        if min_obstacle < 1.0:
            speed_scale = min(speed_scale, 0.15 + 0.35 * min_obstacle)
            speed_scale = max(speed_scale, 0.1)
        speed_scale = float(speed_scale)
        
        local_tx = self.pose[0] + (dx * math.cos(self.pose[2]) - dy * math.sin(self.pose[2]))
        local_ty = self.pose[1] + (dx * math.sin(self.pose[2]) + dy * math.cos(self.pose[2]))
        
        inner_steps = 0
        collision = False
        reached_local = False
        
        while inner_steps < 10 and not collision and not reached_local:
            rclpy.spin_once(self, timeout_sec=self.spin_timeout)
            
            dx_ll = local_tx - self.pose[0]
            dy_ll = local_ty - self.pose[1]
            dist_ll = math.sqrt(dx_ll**2 + dy_ll**2)
            head_err_ll = math.atan2(dy_ll, dx_ll) - self.pose[2]
            head_err_ll = math.atan2(math.sin(head_err_ll), math.cos(head_err_ll))
            
            raw_ll_obs = np.array([
                dist_ll, math.cos(head_err_ll), math.sin(head_err_ll), 
                self.vel[0], self.vel[1], self.vel[2]
            ], dtype=np.float32)
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
            
            msg.linear.x *= speed_scale
            msg.linear.y *= speed_scale
            msg.angular.z *= speed_scale
            
            self.cmd_vel_pub.publish(msg)
            
            if dist_ll < 0.15: 
                reached_local = True
                
            if np.min(self.lidar_data) < 0.45:
                collision = True
                self.cmd_vel_pub.publish(Twist())
                rclpy.spin_once(self, timeout_sec=self.spin_timeout)
                break
            
            inner_steps += 1

        terminated = False
        truncated = False
        reward = -1.0
        using_waypoints = (self.planner is not None and len(self.waypoints) > 0)
        
        if using_waypoints:
            wp1, wp2, wp3 = self._get_lookahead_waypoints()
            dist1, cos1, sin1 = self._calc_target_metrics(wp1)
            dist2, cos2, sin2 = self._calc_target_metrics(wp2)
            dist3, cos3, sin3 = self._calc_target_metrics(wp3)
            
            if dist1 < 0.45:
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.waypoints):
                    reward = 100.0
                    terminated = True
                    collision = False 
                    self.cmd_vel_pub.publish(Twist()) 
                    print(f"✅ GOAL REACHED via waypoints! | Macro-Step: {self.current_step}")
                    dist1 = dist2 = dist3 = 0.0
                else:
                    reward += 5.0
                    self.prev_remaining_path_length = None
                    
                dist_to_global = math.hypot(self.global_goal[0] - self.pose[0], self.global_goal[1] - self.pose[1])
            else:
                current_remaining = self._compute_remaining_path_length()
                if self.prev_remaining_path_length is None:
                    self.prev_remaining_path_length = current_remaining
                progress = self.prev_remaining_path_length - current_remaining
                reward += 3.0 * progress
                self.prev_remaining_path_length = current_remaining
                dist_to_global = current_remaining
            
            is_final = 1.0 
            
        else:
            dist_to_global = math.hypot(self.global_goal[0] - self.pose[0], self.global_goal[1] - self.pose[1])
            if self.prev_dist_to_global is None: self.prev_dist_to_global = dist_to_global
            progress = self.prev_dist_to_global - dist_to_global
            reward += 3.0 * progress
            self.prev_dist_to_global = dist_to_global
            
            dist1, cos1, sin1 = self._calc_target_metrics(self.global_goal)
            dist2, cos2, sin2 = dist1, cos1, sin1
            dist3, cos3, sin3 = dist1, cos1, sin1
            is_final = 1.0

        lidar_min = np.min(self.lidar_data)
        
        if lidar_min < 0.7:
            reward -= 3.0 * speed_scale * (0.7 - lidar_min)
        if lidar_min < 0.5:
            reward -= 10.0      
        if lidar_min > 0.8:
            reward += 1.0 * speed_scale

        if collision and not terminated:
            reward = -100.0
            terminated = True
            wp_info = f"{self.current_wp_idx}/{len(self.waypoints)}" if self.waypoints else "NONE"
            if self.waypoints:
                safe_idx = min(max(0, self.current_wp_idx), len(self.waypoints) - 1)
                wp_coords = self.waypoints[safe_idx]
            else:
                wp_coords = (0, 0)
            mode = "WAYPOINT" if using_waypoints else "DIRECT"
            print(f"🛑 COLLISION! LiDAR Min: {lidar_min:.3f}m | Macro-Step: {self.current_step} | Mode: {mode} | Waypoint: {wp_info}")
        elif self.current_step >= self.max_steps and not terminated:
            truncated = True
            print(f"⏱️ TIMEOUT! | Dist to Goal: {dist_to_global:.2f}m")

        self.csv_writer.writerow([self.current_step, local_tx, local_ty, dist_to_global, speed_scale, reward, int(collision), self.current_wp_idx if using_waypoints else -1])
        self.csv_file.flush()

        # Build 37D State Vector
        robot_state = np.array([
            dist1, cos1, sin1,
            dist2, cos2, sin2,
            dist3, cos3, sin3,
            self.vel[0], self.vel[1], self.vel[2],
            is_final
        ])
        obs = np.concatenate((robot_state, self.lidar_data)).astype(np.float32)
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_dist_to_global = None
        self.prev_remaining_path_length = None
        self.waypoints = []
        self.current_wp_idx = 0
        
        self.cmd_vel_pub.publish(Twist())
        
        teleported = False
        for attempt in range(30):
            if not self.set_state_client.wait_for_service(timeout_sec=1.0): continue
            req = SetEntityState.Request()
            req.state.name = 'my_robot'
            req.state.reference_frame = 'world' 
            req.state.pose.position.x = 0.0
            req.state.pose.position.y = 0.0
            req.state.pose.position.z = 0.05 
            req.state.pose.orientation.w = 1.0 
            
            future = self.set_state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            if future.result() is not None and future.result().success:
                teleported = True
                break
        
        if not teleported:
            self.get_logger().error("CRITICAL: Teleport failed after 30 attempts!")
        
        self.pose[0] = 0.0
        self.pose[1] = 0.0
        self.pose[2] = 0.0 
        self.vel = np.zeros(3)
        self.lidar_data = np.ones(24) * 5.0
        
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05 if self.realtime_eval else 0.01)

        if self.planner is not None:
            max_attempts = 100
            path_found = False
            
            for attempt in range(max_attempts):
                angle = np.random.uniform(-math.pi, math.pi)
                distance = np.random.uniform(2.0, 4.5)
                goal_x = self.pose[0] + distance * math.cos(angle)
                goal_y = self.pose[1] + distance * math.sin(angle)
                gx, gy = self.planner.world_to_grid(goal_x, goal_y)
                
                if not self.planner.is_valid(gx, gy): continue
                
                path = self.planner.plan((self.pose[0], self.pose[1]), (goal_x, goal_y), max_waypoint_spacing=1.5)
                if len(path) >= 2:
                    self.waypoints = path
                    self.current_wp_idx = 1
                    self.global_goal = np.array([goal_x, goal_y])
                    path_found = True
                    break
            
            if not path_found:
                self.waypoints = []
                self.global_goal = np.array([3.0, 0.0])
        else:
            if self.stage in ['1', '2', '3a']:
                while True:
                    angle = np.random.uniform(-math.pi, math.pi)
                    distance = np.random.uniform(2.0, 4.5)
                    goal_x = self.pose[0] + distance * math.cos(angle)
                    goal_y = self.pose[1] + distance * math.sin(angle)
                    if not (-1.0 < goal_x < 1.0 and -1.0 < goal_y < 1.0): break
                self.global_goal = np.array([goal_x, goal_y])
                
            elif self.stage in ['3b', '4']:
                while True:
                    angle = np.random.uniform(-math.pi, math.pi)
                    distance = np.random.uniform(1.5, 3.0)
                    goal_x = self.pose[0] + distance * math.cos(angle)
                    goal_y = self.pose[1] + distance * math.sin(angle)
                    if not (-1.0 < goal_x < 1.0 and -1.0 < goal_y < 1.0): break
                self.global_goal = np.array([goal_x, goal_y])       
            
        using_waypoints = (self.planner is not None and len(self.waypoints) > 0)
        
        if using_waypoints:
            wp1, wp2, wp3 = self._get_lookahead_waypoints()
            dist1, cos1, sin1 = self._calc_target_metrics(wp1)
            dist2, cos2, sin2 = self._calc_target_metrics(wp2)
            dist3, cos3, sin3 = self._calc_target_metrics(wp3)
            is_final = 1.0 
            self.prev_remaining_path_length = self._compute_remaining_path_length()
        else:
            dist1, cos1, sin1 = self._calc_target_metrics(self.global_goal)
            dist2, cos2, sin2 = dist1, cos1, sin1
            dist3, cos3, sin3 = dist1, cos1, sin1
            is_final = 1.0
            self.prev_remaining_path_length = None
        
        robot_state = np.array([
            dist1, cos1, sin1,
            dist2, cos2, sin2,
            dist3, cos3, sin3,
            0.0, 0.0, 0.0, is_final
        ])
        obs = np.concatenate((robot_state, self.lidar_data)).astype(np.float32)
        
        return obs, {}
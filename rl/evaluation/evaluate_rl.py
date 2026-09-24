#!/usr/bin/env python3
#THIS IS EVALUATION SCRIPT FOR PPO, SAC , AND DQN FOR 1ST STUDY.
import rclpy
import numpy as np
import math
import csv
import time
import os
import gymnasium as gym
from datetime import datetime
from std_srvs.srv import Empty 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.environments.continous_tracker_env import OmniRobotTrackerEnv

# UPDATE YOUR PATHS HERE IF NEEDED
MODEL_PATH = "/home/andreas/research/ros_ws/sac_continuous_delta_20260208_225433/sac_tracker_final.zip"
STATS_PATH = "/home/andreas/research/ros_ws/sac_continuous_delta_20260208_225433/vec_normalize.pkl"

class SafeActionWrapper(gym.Wrapper):
    def step(self, action):
        if not isinstance(action, np.ndarray): action = np.array(action)
        action = action.flatten()
        if action.size == 1: action = np.array([float(action[0]), 0.0], dtype=np.float32)
        return self.env.step(action)

def get_errors(goal, pose):
    rx, ry = pose[0], pose[1]
    gx, gy = goal[0], goal[1]
    dx = gx - rx; dy = gy - ry
    dist = math.sqrt(dx**2 + dy**2)
    path_len = math.sqrt(gx**2 + gy**2)
    
    if path_len < 1e-3: return dist, 0.0, 0.0
    
    ux = gx / path_len; uy = gy / path_len
    projection = rx * ux + ry * uy
    lon = path_len - projection
    cross = -rx * uy + ry * ux
    return dist, lon, abs(cross)

def main():
    rclpy.init()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"Results_SAC_5Iterations_{timestamp}.csv"
    f = open(csv_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Time', 'Controller', 'Test_Name', 'Step', 'Goal_X', 'Goal_Y', 
                     'Robot_X', 'Robot_Y', 'Robot_Yaw', 'Euclidean_Err', 'Long_Err', 'Cross_Err'])

    node_tmp = rclpy.create_node('sac_tester_helper')
    reset_client = node_tmp.create_client(Empty, '/reset_simulation')

    raw_env = OmniRobotTrackerEnv()
    wrapped_env = SafeActionWrapper(raw_env)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    if os.path.exists(STATS_PATH):
        vec_env = VecNormalize.load(STATS_PATH, vec_env)
        vec_env.training = False; vec_env.norm_reward = False
        
    model = SAC.load(MODEL_PATH, env=vec_env)

    d = 3.5355
    test_suite = [
        {'name': '1_Forward',   'goal': [5.0, 0.0]},
        {'name': '2_Backward',  'goal': [-5.0, 0.0]},
        {'name': '3_Left',      'goal': [0.0, 5.0]},
        {'name': '4_Right',     'goal': [0.0, -5.0]},
        {'name': '5_Diag_FL',   'goal': [d, d]},
        {'name': '6_Diag_FR',   'goal': [d, -d]},
        {'name': '7_Diag_BL',   'goal': [-d, d]},
        {'name': '8_Diag_BR',   'goal': [-d, -d]},
    ]

    print(f"STARTING SAC EVALUATION (5 Iterations)")

    for iteration in range(5):
        print(f"--- ITERATION {iteration + 1} / 5 ---")
        for test in test_suite:
            if reset_client.service_is_ready(): reset_client.call_async(Empty.Request())
            time.sleep(1.0) 

            vec_env.reset()
            target = np.array(test['goal'])
            raw_env.goal_position = target
            raw_obs = raw_env._get_observation()
            obs = vec_env.normalize_obs(raw_obs)

            print(f"Running Test: {test['name']}")
            start_time = time.time()
            step = 0
            
            while (time.time() - start_time) < 15.0:
                action, _ = model.predict(obs, deterministic=True)
                action_step = np.array([action])
                obs, _, _, _ = vec_env.step(action_step)
                
                rx, ry, ryaw = raw_env.current_pose
                dist, lon, cross = get_errors(test['goal'], [rx, ry, ryaw])
                
                elapsed = time.time() - start_time
                writer.writerow([iteration, elapsed, "RL_Agent", test['name'], step, 
                                 test['goal'][0], test['goal'][1], 
                                 rx, ry, ryaw, dist, lon, cross])
                step += 1

    f.close()
    raw_env.close()
    node_tmp.destroy_node()
    rclpy.shutdown()
    print("SAC Tests Complete.")

if __name__ == "__main__":
    main()
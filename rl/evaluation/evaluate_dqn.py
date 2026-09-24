#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import csv
import time
import os
import glob
import gymnasium as gym
from datetime import datetime
from std_srvs.srv import Empty 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.environments.ddqn_env import OmniRobotDDQNEnv

def get_latest_model_path():
    # Look for folders starting with "rl_discrete_ddqn_" in the current dir
    base_dir = os.getcwd() 
    search_pattern = os.path.join(base_dir, "rl_discrete_ddqn_v1_20260208_061532")
    folders = glob.glob(search_pattern)
    
    if not folders:
        # Fallback to checking a common workspace path if current dir fails
        search_pattern = os.path.join(os.path.expanduser("~"), "rl_discrete_ddqn_v1_20260208_061532")
        folders = glob.glob(search_pattern)
        
    if not folders:
        print("ERROR: No DDQN training folders found! Please check the path.")
        return None, None
    
    # Sort by creation time (newest first)
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"Found latest DDQN training run: {latest_folder}")
    
    # Look for the final model
    model_path = os.path.join(latest_folder, "ddqn_discrete_final.zip")
    if not os.path.exists(model_path):
        # Try alternative naming
        model_files = glob.glob(os.path.join(latest_folder, "ddqn_discrete_*.zip"))
        if model_files:
            model_path = max(model_files, key=os.path.getmtime)
        else:
            print("ERROR: No DDQN model files found in training folder!")
            return None, None
    
    stats = os.path.join(latest_folder, "vec_normalize.pkl")
    return model_path, stats

MODEL_PATH, STATS_PATH = get_latest_model_path()

class DiscreteActionWrapper(gym.Wrapper):
    """Safety wrapper to ensure action is always a clean int for DDQN."""
    def step(self, action):
        if hasattr(action, 'item'):
            safe_action = int(action.item())
        else:
            safe_action = int(action)
        return self.env.step(safe_action)

def get_errors(goal, pose):
    """Calculate distance and alignment errors."""
    ex, ey = goal[0] - pose[0], goal[1] - pose[1]
    yaw = pose[2]
    dist = math.sqrt(ex**2 + ey**2)
    lon = ex * math.cos(yaw) + ey * math.sin(yaw)
    cross = -ex * math.sin(yaw) + ey * math.cos(yaw)
    return dist, lon, cross

def main():
    rclpy.init()
    
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"Results_DDQN_{timestamp}.csv"
    f = open(csv_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Time', 'Controller', 'Test_Name', 'Step', 'Goal_X', 'Goal_Y', 
                     'Robot_X', 'Robot_Y', 'Robot_Yaw', 'Euclidean_Err', 'Long_Err', 'Cross_Err'])

    node_tmp = rclpy.create_node('ddqn_tester_helper')
    reset_client = node_tmp.create_client(Empty, '/reset_simulation')

    # --- 2. SETUP ENVIRONMENT & NORMALIZATION ---
    raw_env = OmniRobotDDQNEnv() 
    wrapped_env = DiscreteActionWrapper(raw_env)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    # Load Normalization Stats (Crucial for Agent Vision)
    if os.path.exists(STATS_PATH):
        print(f"Loading normalization stats from: {STATS_PATH}")
        vec_env = VecNormalize.load(STATS_PATH, vec_env)
        vec_env.training = False     # Do not update stats during test
        vec_env.norm_reward = False  # Do not normalize reward output
    else:
        print("WARNING: vec_normalize.pkl NOT FOUND. Agent may not perform optimally.")
        
    model = DQN.load(MODEL_PATH, env=vec_env)

    # 8 Directions test suite
    d = 3.5355  # 5.0m diagonal
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

    print(f"STARTING DDQN EVALUATION")
    print(f"Logging to: {csv_name}")

    for test in test_suite:
        # Reset Simulation
        if reset_client.service_is_ready():
            reset_client.call_async(Empty.Request())
        time.sleep(1.0) 

        # Reset Env Wrapper
        vec_env.reset()
        
        target = np.array(test['goal'])
        raw_env.goal_position = target
        
        raw_obs = raw_env._get_observation()
        obs = vec_env.normalize_obs(raw_obs)

        print(f"Running Test: {test['name']} -> Goal: {target}")
        
        start_time = time.time()
        step = 0
        
        # --- 3. EVALUATION LOOP ---
        while (time.time() - start_time) < 25.0:
            action, _ = model.predict(obs, deterministic=True)
            
            action_step = np.array([action])
            
            # Step returns: obs, reward, done, info
            obs, _, dones, _ = vec_env.step(action_step)
            
            rx, ry, ryaw = raw_env.current_pose
            dist, lon, cross = get_errors(test['goal'], [rx, ry, ryaw])
            
            elapsed = time.time() - start_time
            writer.writerow([elapsed, "DDQN_Agent", test['name'], step, 
                             test['goal'][0], test['goal'][1], 
                             rx, ry, ryaw, 
                             dist, lon, cross])
            step += 1
            
            # CHECK IF GOAL REACHED
            if dones[0]:
                print(f"SUCCESS! Goal reached in {elapsed:.2f}s. Final Dist: {dist:.4f}m")
                # Break the loop immediately so we don't log the reset position
                break

    f.close()
    raw_env.close()
    node_tmp.destroy_node()
    rclpy.shutdown()
    print("DDQN Evaluation Complete.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import glob
import csv
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Low_level_rl import LieGroupLowLevelEnv

# --- AUTO-DETECT LATEST MODEL PATH ---
def get_latest_model_paths():
    base_dir = os.getcwd() 
    search_path = os.path.join(base_dir, "/home/andreas/research/ros_ws/ll_motor_models_20260310_212843") 
    folders = glob.glob(search_path)
    
    if not folders:
        print("ERROR: No tracking models found! Please check the path.")
        return None, None
    
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"Found latest tracking run: {latest_folder}")
    
    model_file = os.path.join(latest_folder, "ll_motor_driver_final")
    stats_file = os.path.join(latest_folder, "vec_normalize.pkl")
    return model_file, stats_file

def main():
    model_path, stats_path = get_latest_model_paths()
    if not model_path or not os.path.exists(stats_path):
        print("ERROR: Could not find model or vec_normalize.pkl!")
        return

    print(f"Loading New LL Tracker from: {model_path}")

    raw_env = LieGroupLowLevelEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False 
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    
    # CSV Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"eval_new_tracker_{timestamp}.csv"
    
    print("\nStarting Evaluation of Low_level_rl.py... Watch Gazebo!")
    print(f"Logging data to {csv_filename}")
    print(f"{'Step':<6} | {'Target (vx, vy, wz)':<30} | {'Actual (vx, vy, wz)':<30}")
    print("-" * 75)
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Step', 'Target_Vx', 'Target_Vy', 'Target_Wz', 'Actual_Vx', 'Actual_Vy', 'Actual_Wz', 'Reward'])
        
        for step in range(1500):
            action, _states = model.predict(obs, deterministic=True)
            
            if not isinstance(action, np.ndarray) or len(action.shape) == 1:
                action_step = np.array([action])
            else:
                action_step = action
                
            obs, rewards, dones, infos = vec_env.step(action_step)
            
            target = raw_env.target_twist
            actual_vx = raw_env.node.actual_vx
            actual_vy = raw_env.node.actual_vy
            actual_wz = raw_env.node.actual_wz
            
            # Log to CSV
            writer.writerow([
                step, 
                target[0], target[1], target[2],
                actual_vx, actual_vy, actual_wz, 
                rewards[0]
            ])
            
            if step % 20 == 0:
                target_str = f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
                actual_str = f"({actual_vx:.2f}, {actual_vy:.2f}, {actual_wz:.2f})"
                print(f"{step:<6} | {target_str:<30} | {actual_str:<30}")

            if dones[0]:
                print("\n--- Generating new random twist ---\n")
                obs = vec_env.reset()
                
    print(f"\nEvaluation complete. Data saved to {csv_filename}")

if __name__ == '__main__':
    main()
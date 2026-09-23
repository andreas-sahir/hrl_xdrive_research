#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import pandas as pd
import time
import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

# Import your environment (Stages 1-3 version)
from hrl_env123 import HRLNavEnv

def calc_local_errors(start_pose, local_target, current_pose):
    """Calculates Local Euclidean, Longitudinal, and Cross-Track Errors."""
    rx, ry = current_pose[0], current_pose[1]
    sx, sy = start_pose[0], start_pose[1]
    gx, gy = local_target[0], local_target[1]
    
    euc_err = math.sqrt((gx - rx)**2 + (gy - ry)**2)
    
    px, py = gx - sx, gy - sy
    path_len = math.sqrt(px**2 + py**2)
    
    if path_len < 1e-6: 
        return euc_err, euc_err, 0.0
        
    ux, uy = px/path_len, py/path_len
    rx_vec, ry_vec = rx - sx, ry - sy
    
    lon_progress = rx_vec * ux + ry_vec * uy
    lon_err = path_len - lon_progress
    cross_err = abs(rx_vec * uy - ry_vec * ux)
    
    return euc_err, lon_err, cross_err

def main():
    rclpy.init()
    
    # ==========================================
    # 1. CONFIG — UPDATE THESE PATHS
    # ==========================================
    TEST_STAGE = '3b'  # Change to '1', '2', '3a', or '3b' as needed

    # Low-Level Weights (Frozen)
    LL_MODEL_PATH = "/home/andreas/research/ros_ws/rl_discrete_ppo_v5_20260206_222937/ppo_discrete_final.zip"
    LL_STATS_PATH = "/home/andreas/research/ros_ws/rl_discrete_ppo_v5_20260206_222937/vec_normalize.pkl"
    
    # High-Level Checkpoint (Ensure this matches the TEST_STAGE!)
    HL_MODEL_PATH = "/home/andreas/research/ros_ws/retrain (no waypoints)/hrl_sac_s3a_20260627_052218/stage3_master.zip"
    HL_STATS_PATH = "/home/andreas/research/ros_ws/retrain (no waypoints)/hrl_sac_s3a_20260627_052218/vec_normalize_stage3.pkl"
    
    NUM_EPISODES = 50
    
    print("\n" + "="*60)
    print(f"STAGE {TEST_STAGE} METRICS EVALUATION PROTOCOL (Deterministic Mode)")
    print("="*60)
    
    # ==========================================
    # 2. INITIALIZE ENVIRONMENT STACK
    # ==========================================
    raw_env = HRLNavEnv(
        ll_model_path=LL_MODEL_PATH, 
        ll_stats_path=LL_STATS_PATH, 
        stage=TEST_STAGE,
        realtime_eval=True
    )
    
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    print(f"Loading HL Stats from: {HL_STATS_PATH}")
    env = VecNormalize.load(HL_STATS_PATH, vec_env)
    env.training = False 
    env.norm_reward = False

    print(f"Loading HL Brain from: {HL_MODEL_PATH}")
    model = SAC.load(HL_MODEL_PATH, env=env)

    # ==========================================
    # 3. EVALUATION LOOP
    # ==========================================
    metrics_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"evaluation_stage{TEST_STAGE}_metrics_{timestamp}.csv"
    
    print("\nRunning Evaluation Trajectories... Watch Gazebo Footages closely!")
    print("-" * 60)
    
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        step = 0
        
        ep_ctes, ep_lons, ep_eucs = [], [], []
        
        raw_node = env.venv.envs[0]
        
        while not done:
            start_pos = [raw_node.pose[0], raw_node.pose[1]]
            start_yaw = raw_node.pose[2]
            
            # The only true target in Stages 1-3 is the direct global goal
            local_target = [raw_node.global_goal[0], raw_node.global_goal[1]]
            
            # Predict actions cleanly without entropy swerves
            action, _states = model.predict(obs, deterministic=True)
            
            # Handle Gymnasium 5-tuple packing inside SB3 vector format 
            obs, reward, dones, infos = env.step(action)
            
            done = dones[0]
            step += 1
            
            end_pos = [raw_node.pose[0], raw_node.pose[1]]
            
            # Calculate Kinematic Drift Metrics
            euc_err, lon_err, cross_err = calc_local_errors(start_pos, local_target, end_pos)
            ep_eucs.append(euc_err)
            ep_lons.append(lon_err)
            ep_ctes.append(cross_err)
            
            # Frame bounding delay to match 1:1 simulation updates
            time.sleep(0.01)

        # Evaluate final state output logic
        avg_cte = np.mean(ep_ctes) if ep_ctes else 0.0
        max_cte = np.max(ep_ctes) if ep_ctes else 0.0
        avg_lon = np.mean(ep_lons) if ep_lons else 0.0
        avg_euc = np.mean(ep_eucs) if ep_eucs else 0.0
        
        # Robust Success Identification via Spatial Boundaries
        # For Stages 1-3, if it reaches the goal, it triggers reward = 100.0 and terminated = True
        has_collided = (np.min(raw_node.lidar_data) < 0.45)
        is_timeout = (step >= raw_node.max_steps)
        
        if not has_collided and not is_timeout:
            status = "Success"
        else:
            status = "Crash/Timeout"

        metrics_data.append({
            "Episode": ep + 1,
            "Status": status,
            "Steps": step,
            "Avg_CTE_m": round(avg_cte, 4),
            "Max_CTE_m": round(max_cte, 4),
            "Avg_Lon_Error_m": round(avg_lon, 4),
            "Avg_Euc_Error_m": round(avg_euc, 4)
        })
        
        print(f"Ep {ep+1:<3} | {status:<13} | Steps: {step:<3} | CTE: {avg_cte:.4f}m")
        
        # Stabilizing delay for telemetry resets
        time.sleep(0.5)

    # ==========================================
    # 4. DATA COMPILATION & SAVE
    # ==========================================
    df = pd.DataFrame(metrics_data)
    df.to_csv(csv_filename, index=False)
    
    success_rate = (len(df[df['Status'] == 'Success']) / NUM_EPISODES) * 100
    
    print("\n" + "=" * 60)
    print("EVALUATION PROTOCOL COMPLETED")
    print("=" * 60)
    print(f"Success Rate            : {success_rate:.1f}%")
    print(f"Overall Avg Local CTE   : {df['Avg_CTE_m'].mean():.4f} m")
    print(f"Overall Max Local CTE   : {df['Max_CTE_m'].mean():.4f} m")
    print(f"Overall Avg Long. Error : {df['Avg_Lon_Error_m'].mean():.4f} m")
    print(f"Overall Avg Euc. Error  : {df['Avg_Euc_Error_m'].mean():.4f} m")
    print(f"Dataset saved to        : {csv_filename}")
    print("=" * 60)

if __name__ == "__main__":
    main()
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

from hrl_env import HRLNavEnv

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
    # CONFIGURATION
    # ==========================================
    LL_MODEL_PATH = "/home/andreas/research/ros_ws/rl_discrete_ppo_v5_20260206_222937/ppo_discrete_final.zip"
    LL_STATS_PATH = "/home/andreas/research/ros_ws/rl_discrete_ppo_v5_20260206_222937/vec_normalize.pkl"
    
    HL_MODEL_PATH = "/home/andreas/research/ros_ws/retrain (no waypoints)/hrl_sac_s4_fresh_20260717_170006/stage4_master.zip"
    HL_STATS_PATH = "/home/andreas/research/ros_ws/retrain (no waypoints)/hrl_sac_s4_fresh_20260717_170006/vec_normalize_stage4.pkl"
    MAP_PATH = "/home/andreas/research/ros_ws/src/my_robot/maps/stage4_map1.pgm"
    
    NUM_EPISODES = 50
    
    print("\n" + "="*60)
    print("STAGE 4 METRICS & A* TRAJECTORY EVALUATION")
    print("="*60)
    
    raw_env = HRLNavEnv(
        ll_model_path=LL_MODEL_PATH, 
        ll_stats_path=LL_STATS_PATH, 
        stage='4',
        realtime_eval=True,
        map_path=MAP_PATH
    )
    
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    env = VecNormalize.load(HL_STATS_PATH, vec_env)
    env.training = False 
    env.norm_reward = False

    model = SAC.load(HL_MODEL_PATH, env=env)

    metrics_data = []
    trajectory_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for ep in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        step = 0
        ep_ctes, ep_lons, ep_eucs = [], [], []
        raw_node = env.venv.envs[0]
        
        astar_path_str = ";".join([f"{wp[0]},{wp[1]}" for wp in raw_node.waypoints]) if raw_node.waypoints else ""
        last_reward = 0.0
        
        while not done:
            start_pos = [raw_node.pose[0], raw_node.pose[1]]
            start_yaw = raw_node.pose[2]
            
            action, _states = model.predict(obs, deterministic=True)
            act = action[0]
            
            local_tx = start_pos[0] + (act[0] * math.cos(start_yaw) - act[1] * math.sin(start_yaw))
            local_ty = start_pos[1] + (act[0] * math.sin(start_yaw) + act[1] * math.cos(start_yaw))
            local_target = [local_tx, local_ty]
            
            obs, reward, dones, infos = env.step(action)
            last_reward = reward[0]
            done = dones[0]
            step += 1
            
            if done and 'terminal_observation' in infos[0]:
                end_pos = local_target
            else:
                end_pos = [raw_node.pose[0], raw_node.pose[1]]
            
            euc_err, lon_err, cross_err = calc_local_errors(start_pos, local_target, end_pos)
            ep_eucs.append(euc_err)
            ep_lons.append(lon_err)
            ep_ctes.append(cross_err)
            
            trajectory_data.append({
                "Episode": ep + 1,
                "Step": step,
                "Robot_X": end_pos[0],
                "Robot_Y": end_pos[1],
                "Global_Goal_X": raw_node.global_goal[0],
                "Global_Goal_Y": raw_node.global_goal[1],
                "AStar_Path": astar_path_str if step == 1 else "" 
            })
            time.sleep(0.01)

        avg_cte = np.mean(ep_ctes) if ep_ctes else 0.0
        max_cte = np.max(ep_ctes) if ep_ctes else 0.0
        avg_lon = np.mean(ep_lons) if ep_lons else 0.0
        avg_euc = np.mean(ep_eucs) if ep_eucs else 0.0
        
        # Determine status strictly by the final reward issued by hrl_env.py
        if last_reward <= -50.0:
            status = "Crash"
        elif last_reward >= 50.0:
            status = "Success"
        else:
            status = "Timeout"

        # Matching exact target metric columns
        metrics_data.append({
            "Episode": ep + 1,
            "Status": status,
            "Steps": step,
            "Avg_CTE_m": round(avg_cte, 4),
            "Max_CTE_m": round(max_cte, 4),
            "Avg_Lon_Error_m": round(avg_lon, 4),
            "Avg_Euc_Error_m": round(avg_euc, 4)
        })
        
        print(f"Ep {ep+1:<3} | {status:<10} | Steps: {step:<3} | CTE: {avg_cte:.4f}m | Lon: {avg_lon:.4f}m | Term Reward: {last_reward:.1f}")
        time.sleep(0.5)

    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(f"eval_s4_metrics_{timestamp}.csv", index=False)
    pd.DataFrame(trajectory_data).to_csv(f"eval_s4_trajectories_{timestamp}.csv", index=False)
    
    success_rate = (len(df_metrics[df_metrics['Status'] == 'Success']) / NUM_EPISODES) * 100
    print("\n" + "=" * 60)
    print("EVALUATION PROTOCOL COMPLETED")
    print("=" * 60)
    print(f"True Success Rate       : {success_rate:.1f}%")
    print(f"Overall Avg Local CTE   : {df_metrics['Avg_CTE_m'].mean():.4f} m")
    print(f"Overall Max Local CTE   : {df_metrics['Max_CTE_m'].mean():.4f} m")
    print(f"Overall Avg Long. Error : {df_metrics['Avg_Lon_Error_m'].mean():.4f} m")
    print(f"Overall Avg Euc. Error  : {df_metrics['Avg_Euc_Error_m'].mean():.4f} m")
    print("=" * 60)

if __name__ == "__main__":
    main()
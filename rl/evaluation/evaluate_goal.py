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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.environments.omni_rl_env import OmniRobotRLController

# ------------------------------------------------------------------
# MODEL DISCOVERY AND EVALUATION SETUP
# This helper automatically locates the latest trained checkpoint so the
# evaluator can load the model and its normalization statistics without
# needing a hard-coded path for every run.
# ------------------------------------------------------------------
def get_latest_model_path():
    base_dir = os.getcwd()
    # NOTE: You might need to adjust this search string if you trained a new model!
    search_path = os.path.join(base_dir, "rl_discrete_ppo_v5_20260206_222937")
    folders = glob.glob(search_path)
    
    if not folders:
        search_path = os.path.join(os.path.expanduser("~"), "rl_discrete_ppo_v5_20260206_222937")
        folders = glob.glob(search_path)
        
    if not folders:
        print("ERROR: No training folders found! Please check the path.")
        return None, None
    
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"Found latest training run: {latest_folder}")
    
    model = os.path.join(latest_folder, "ppo_discrete_final.zip")
    stats = os.path.join(latest_folder, "vec_normalize.pkl")
    return model, stats

MODEL_PATH, STATS_PATH = get_latest_model_path()

class DiscreteShimWrapper(gym.Wrapper):
    def step(self, action):
        if hasattr(action, 'item'): safe_action = int(action.item())
        else: safe_action = int(action)
        return self.env.step(safe_action)

def get_errors(goal, pose):
    # ------------------------------------------------------------------
    # ERROR METRICS
    # The evaluator decomposes tracking quality into (1) Euclidean distance,
    # (2) longitudinal error along the ideal path, and (3) cross-track error
    # perpendicular to that path. These values are useful for diagnosing whether
    # the robot is missing the target, drifting sideways, or progressing well.
    # ------------------------------------------------------------------
    rx, ry = pose[0], pose[1]
    gx, gy = goal[0], goal[1]

    # 1. Euclidean Distance
    dx = gx - rx; dy = gy - ry
    dist = math.sqrt(dx**2 + dy**2)

    # 2. Path Vector
    path_len = math.sqrt(gx**2 + gy**2)
    if path_len < 1e-3: return dist, 0.0, 0.0

    # Unit Vector of the Ideal Path
    ux = gx / path_len; uy = gy / path_len

    # 3. Project Robot Position onto the Ideal Path
    projection = rx * ux + ry * uy

    # 4. Longitudinal Error (distance remaining along the path)
    lon = path_len - projection

    # 5. Cross-Track Error (perpendicular distance from the path)
    cross = -rx * uy + ry * ux

    return dist, lon, abs(cross)

def main():
    rclpy.init()
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"Results_PPO_5Iterations_{timestamp}.csv"
    f = open(csv_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Time', 'Controller', 'Test_Name', 'Step', 'Goal_X', 'Goal_Y', 
                     'Robot_X', 'Robot_Y', 'Robot_Yaw', 'Euclidean_Err', 'Long_Err', 'Cross_Err'])

    node_tmp = rclpy.create_node('ppo_tester_helper')
    reset_client = node_tmp.create_client(Empty, '/reset_simulation')

    raw_env = OmniRobotRLController() 
    wrapped_env = DiscreteShimWrapper(raw_env)
    vec_env = DummyVecEnv([lambda: wrapped_env])
    
    if os.path.exists(STATS_PATH):
        print(f"Loading normalization stats from: {STATS_PATH}")
        vec_env = VecNormalize.load(STATS_PATH, vec_env)
        vec_env.training = False; vec_env.norm_reward = False
        
    model = PPO.load(MODEL_PATH, env=vec_env)

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

    print(f"STARTING PPO EVALUATION (5 Iterations)")

    # ------------------------------------------------------------------
    # OUTER LOOP FOR FIVE REPEATED TRIALS
    # Repeating the same test several times reduces the chance that a single
    # lucky trajectory masks a systematic control issue.
    # ------------------------------------------------------------------
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
            
            while (time.time() - start_time) < 25.0:
                action, _ = model.predict(obs, deterministic=True)
                action_step = np.array([action])
                obs, _, dones, _ = vec_env.step(action_step)
                
                rx, ry, ryaw = raw_env.current_pose
                dist, lon, cross = get_errors(test['goal'], [rx, ry, ryaw])
                
                elapsed = time.time() - start_time
                writer.writerow([iteration, elapsed, "RL_Agent", test['name'], step, 
                                 test['goal'][0], test['goal'][1], 
                                 rx, ry, ryaw, dist, lon, cross])
                step += 1
                
                if dones[0]:
                    print(f"SUCCESS! {test['name']} finished in {elapsed:.2f}s.")
                    break

    f.close()
    raw_env.close()
    node_tmp.destroy_node()
    rclpy.shutdown()
    print("PPO Evaluation Complete.")

if __name__ == "__main__":
    main()
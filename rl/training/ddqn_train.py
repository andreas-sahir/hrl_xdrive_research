#!/usr/bin/env python3
import rclpy
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime

# Import the DDQN environment
from ddqn_env import OmniRobotDDQNEnv

def main():
    rclpy.init()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"rl_discrete_ddqn_v1_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"STARTING DISCRETE DDQN TRAINING (Anti-Fall/Spinning V1)")
    print(f"Logs: {log_dir}")
    
    env_node = None
    try:
        # 1. Create Environment
        env_node = OmniRobotDDQNEnv()
        env = Monitor(env_node, log_dir)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # 2. Setup Model with PPO hyperparameters adapted for DDQN
        model = DQN(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=0.0003,      # Same as PPO
            batch_size=256,              # Same as PPO
            buffer_size=200000,         # DDQN specific
            train_freq=8,               # DDQN specific
            target_update_interval=10000, # DDQN specific
            gamma=0.99,                 # Same as PPO
            exploration_fraction=0.1,   # Exploration schedule
            exploration_initial_eps=1.0, # Start with full exploration
            exploration_final_eps=0.05,  # End with small exploration
            tensorboard_log=log_dir,
            device="cuda"                # DDQN can be faster on CPU for discrete envs
        )
        
        # 3. Train
        print("Training... (Press Ctrl+C to stop)")
        model.learn(total_timesteps=1000000, tb_log_name="DQN_Discrete_V1")
        
        # 4. Save
        model.save(f"{log_dir}/ddqn_discrete_final")
        vec_env.save(f"{log_dir}/vec_normalize.pkl")
        print("Model Saved.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if env_node:
            env_node.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
import os
from datetime import datetime

# Import the UPDATED CONTINUOUS environment
from tracker_rl import OmniRobotTrackerEnv

def main():
    rclpy.init()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Updated naming to match PPO
    log_dir = f"sac_continuous_delta_{timestamp}" 
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"STARTING SAC TRAINING (Delta Plan: 1.0 m/s, Friction 0.2)")
    print(f"Logs: {log_dir}")
    
    env_node = None
    try:
        # 1. Create Environment
        env_node = OmniRobotTrackerEnv()
        env = Monitor(env_node, log_dir)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # 2. Model Setup
        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=100000, 
            batch_size=512,
            ent_coef='auto',
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda" 
        )
        model.set_logger(logger)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=log_dir,
            name_prefix="sac_tracker"
        )
        
        print("Training... (Press Ctrl+C to stop)")
        
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback,
            tb_log_name="SAC_Delta_1M",
            log_interval=4  # Print training stats (FPS) every 4 episodes
        )
        
        model.save(f"{log_dir}/sac_tracker_final")
        vec_env.save(f"{log_dir}/vec_normalize.pkl")
        print("Training Complete. Model Saved.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        if 'model' in locals():
            model.save(f"{log_dir}/sac_tracker_interrupted")
            vec_env.save(f"{log_dir}/vec_normalize.pkl")
    finally:
        if env_node:
            env_node.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
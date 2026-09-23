#!/usr/bin/env python3

import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Low_level_rl import LieGroupLowLevelEnv

# --- CUSTOM LOGGING CALLBACK ---
class StepLoggerCallback(BaseCallback):
    def __init__(self, print_freq: int, verbose=1):
        super(StepLoggerCallback, self).__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # Check if we hit the 100-step mark
        if self.n_calls % self.print_freq == 0:
            # Get the reward from the current step
            reward = self.locals['rewards'][0]
            
            # Extract the custom info dictionary from our environment
            info = self.locals['infos'][0]
            err_vx = info.get('error_vx', 0.0)
            err_vy = info.get('error_vy', 0.0)
            err_wz = info.get('error_wz', 0.0)
            
            # Print a clean, formatted log to the terminal
            print(f"Step: {self.n_calls:<6} | "
                  f"Reward: {reward:>7.3f} | "
                  f"Error (Vx: {err_vx:>6.3f}, Vy: {err_vy:>6.3f}, Wz: {err_wz:>6.3f})")
            
        return True
# -------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./ll_motor_logs_{timestamp}/"
    model_dir = f"./ll_motor_models_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("STARTING CONTINUOUS PPO TRAINING (Stage 1: Low-Level Controller)")
    
    # 1. Initialize and wrap the environment
    raw_env = LieGroupLowLevelEnv()
    check_env(raw_env, warn=True)
    
    env = Monitor(raw_env, log_dir)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 2. Setup the PPO Agent
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, 
        verbose=0, # Set to 0 so the default SB3 table doesn't spam your custom logs
        tensorboard_log=log_dir,
        device="cpu" 
    )
    
    # 3. Setup Callbacks
    eval_callback = EvalCallback(
        vec_env, 
        best_model_save_path=model_dir,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Instantiate our custom logger to print every 100 steps
    step_logger = StepLoggerCallback(print_freq=100)
    
    # Combine both callbacks into a list
    callback_list = CallbackList([eval_callback, step_logger])
    
    # 4. Start Training
    total_timesteps = 1_000_000 
    
    try:
        print(f"{'='*60}\nTraining... (Press Ctrl+C to stop and save)\n{'='*60}")
        model.learn(total_timesteps=total_timesteps, callback=callback_list, tb_log_name="PPO_Continuous_LL")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current progress...")
    finally:
        final_model_path = os.path.join(model_dir, "ll_motor_driver_final")
        model.save(final_model_path)
        vec_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
        print(f"Low-Level Model saved to {model_dir}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
from hrl_env import HRLNavEnv

def main():
    rclpy.init()
    
    # ------------------------------------------------------------------
    # LOW-LEVEL POLICY INTEGRATION
    # The high-level controller is trained on top of a frozen low-level policy
    # that already handles local drift correction. Loading the fixed model and
    # normalizer keeps the hierarchical stack stable while the meta-policy
    # learns route selection and waypoint tracking.
    # ------------------------------------------------------------------
    LL_MODEL_PATH = "/home/andreas/research/ros_ws/rl_discrete_ppo_v5_20260206_222937/ppo_discrete_final.zip"
    LL_STATS_PATH = "/home/andreas/research/ros_ws/rl_discrete_ppo_v5_20260206_222937/vec_normalize.pkl"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"hrl_sac_s4_hybrid_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING HRL HIGH-LEVEL TRAINING FOR STAGE 4 (Hybrid SMDP)")
    print(f"Logs: {log_dir}")
    print("=" * 60)

    MAP_PATH = "/home/andreas/research/ros_ws/src/my_robot/maps/stage4_map1.pgm"
    env_node = HRLNavEnv(
        ll_model_path=LL_MODEL_PATH,
        ll_stats_path=LL_STATS_PATH,
        stage='4',
        map_path=MAP_PATH
    )

    env_node = Monitor(env_node, log_dir)
    vec_env = DummyVecEnv([lambda: env_node])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # ------------------------------------------------------------------
    # TRAINING NORMALIZATION AND ARCHITECTURE
    # The state normalization is reset for this high-level run so the policy is
    # trained against a clean distribution. The SAC agent then learns a macro
    # action policy that chooses motion commands based on global planning state.
    # ------------------------------------------------------------------
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env.training = True

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=300000,
        batch_size=512,
        ent_coef='auto',
        target_entropy='auto',
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda"
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=log_dir, name_prefix="hrl_sac")

    try:
        model.learn(total_timesteps=1500000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually!")
    finally:
        final_model_path = os.path.join(log_dir, "stage4_master.zip")
        final_stats_path = os.path.join(log_dir, "vec_normalize_stage4.pkl")
        model.save(final_model_path)
        vec_env.save(final_stats_path)
        print(f"\nModel saved to: {final_model_path}")

if __name__ == "__main__":
    main()
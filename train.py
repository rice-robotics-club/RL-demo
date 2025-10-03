# This script sets up a PyBullet-based reinforcement learning environment for a
# simple quadruped robot. The task is to reach and touch a green target box
# (specified by its center and size) within one episode (~30s by default),
# while staying upright and avoiding jumping.
#
# Dependencies:
#   pip install pybullet gymnasium stable-baselines3[extra]
#
# Notes:
# - The URDF file 'simple_quadruped.urdf' must be in the same directory.
# - Uses Stable-Baselines3 PPO as the baseline RL agent.

import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# These are our custom modules: 
# utils has a bunch of helper functions for importing stuff and navigating local files
# env has our custom BaseEnv envrironment class that inherits from gym.Env and implements the RL environment
from src.envs import env
from src.utils import utils

if __name__ == "__main__":

    ## NO MORE HARD-CODED RENDER MODES! JUST SET IT AT RUNTIME!! YIPPEEEEE!!!!! ## 
    user_render_mode_request = input("Run with GUI? (y/n): ").strip().lower()
    if user_render_mode_request == 'y':
        render_mode = 'human'
    else:
        render_mode = 'headless'
    
    urdf_file, save_path, save_prefix = utils.select_robot(load_model=False)

    # Pass box parameters into the environment.
    min_z = env.get_min_z(urdf_file)
    env = env.BaseEnv(
        render_mode=render_mode, 
        urdf_filename=urdf_file, 
        start_position=[0, 0, -min_z],
        target_speed = .5,
    )
    use_existing_model = input("Use existing model if available? (y/n): ").strip().lower() == 'y'
    if use_existing_model:
        model_directory_list = os.listdir(save_path)
        print("Model Directory List: ", model_directory_list)
    
        # Handle case with no models found
        if len(model_directory_list) == 0:
            print("No saved models found. A new model will be created.")
            use_existing_model = False
        else:
            best_model_name = model_directory_list[-1]
            best_model_path = os.path.join(save_path, best_model_name)
            
            # Handle case with multiple models found
            if len(model_directory_list) > 1:
                print("Multiple saved models found. Would you like to use the latest one? (y/n): ")
                choice = input().strip().lower()
                if choice != 'y':
                    print("Available models:")
                    for idx, model_name in enumerate(model_directory_list):
                        print(f"{idx + 1}: {model_name}")
                    print("Enter the number of the model you want to use: ")
                    selected_idx = int(input().strip()) - 1
                    if 0 <= selected_idx < len(model_directory_list):
                        best_model_name = model_directory_list[selected_idx]
                        best_model_path = os.path.join(save_path, best_model_name)
                    else:
                        print("Invalid selection. Using the latest model.")

            print(f"Loading existing model from {best_model_path}")
            model = PPO.load(best_model_path, env=env, device='cpu')
    else:
        model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)  # Slightly larger n_steps may help with harder tasks

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_path,
        name_prefix=save_prefix
    )
    
    try:

        model.learn(total_timesteps=2000000, callback=checkpoint_callback)  # This task may require longer training
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")

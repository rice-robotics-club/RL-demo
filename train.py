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

    # Set target box center [x, y] and size [width, depth, height].
    box_center = [12.0, 3.0]
    box_size = [2.0, 2.0, 1.0]  # A 2x2x1 m box

    # Pass box parameters into the environment.
    min_z = env.get_min_z(urdf_file)
    env = env.BaseEnv(
        render_mode=render_mode, 
        urdf_filename=urdf_file, 
        start_position=[0, 0, -min_z],
        target_box_center=box_center,
        target_box_size=box_size
    )
    
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)  # Slightly larger n_steps may help with harder tasks

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_path,
        name_prefix=save_prefix
    )
    
    print(f"Starting training... Target Box Center: {box_center}, Size: {box_size}")
    try:

        model.learn(total_timesteps=1000000, callback=checkpoint_callback)  # This task may require longer training
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")

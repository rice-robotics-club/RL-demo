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
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path

from src.envs import env

def main():
    parser = argparse.ArgumentParser("Script for training Quadruped RL policy in sim.")

    parser.add_argument("-m", "--model_path", type=str, help="model file to load for training")
    parser.add_argument("--headless", default=False, action="store_true", help="disables the simulator GUI")
    parser.add_argument("-u", "--urdf_path", type=str, default="./robots/full_servobot/servobot.urdf", help="path to robot URDF file")
    parser.add_argument("-p", "--prefix", type=str, default="servobot", help="prefix to use for model files")
    parser.add_argument("-s", "--save_path", type=str, default="./models/", help="directory path to save model files to")

    args = parser.parse_args()

    min_z = env.get_min_z(args.urdf_path)
    environment = env.BaseEnv(
        headless=args.headless,
        urdf_filename=args.urdf_path,
        start_position=[0, 0, -min_z],
        target_speed=.5,
    )

    if args.model_path:
        model = PPO.load(args.model_path, env=environment, device='cpu')
    else:
        model = PPO("MlpPolicy", env=environment, verbose=1, n_steps=2048)  # Slightly larger n_steps may help with harder tasks

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=args.save_path,
        name_prefix=args.prefix
    )

    try:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback)  # This task may require longer training
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        environment.close()

    print("Training finished.")


if __name__ == "__main__":
    main()

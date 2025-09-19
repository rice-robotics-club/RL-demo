# This script sets up a reinforcement learning environment to train a simple quadruped robot
# to move as far as possible in 10 seconds. It uses the PyBullet physics engine for the
# simulation and the stable-baselines3 library for the RL agent.
#
# To run this script, you'll need to install the required libraries:
# pip install pybullet gymnasium stable-baselines3[extra]
#
# The script now assumes that 'simple_quadruped.urdf' exists in the same directory.

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Our custom environment class
from src.env import QuadrupedEnv

if __name__ == "__main__":
    # To use a different robot, change the filename here
    urdf_file = "full_servobot/catbot.urdf" 
    # Create the environment. Stable-baselines will automatically call reset.
    env = QuadrupedEnv(render_mode='human', urdf_filename=urdf_file)
    
    # Define the PPO agent from stable-baselines3
    model = PPO("MlpPolicy", env, verbose=1,n_steps=1024)

    # Setup Checkpoint Callback to save the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./servobot_checkpoints/',
        name_prefix='servobot_model'
    )
    
    # Train the agent for a number of timesteps
    print("Starting training...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

from src.envs.env import BaseEnv
from src.utils import utils

'''
This script is a mismash of the quadreped run_trained.py and the servobot train.py scripts to load and run a trained servobot model
to demonstrate its movement in the pybullet gui. right now it looks pretty silly bc i just ran the training on my laptop for 15 minutes
but we could prolly get some better results if we ran it longer. 
'''

if __name__ == "__main__":
    # To use a different robot, change the filename here

    urdf_file, save_path, save_prefix, model_path = utils.select_robot()

    # Create the environment. Stable-baselines will automatically call reset.
    env = BaseEnv(render_mode='human', urdf_filename=urdf_file)

    # check to make sure we didnt forget to import a model lmao
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train your model first using train_quadruped.py.")
    else:
        # Load the trained model
        model = PPO.load(model_path, env=env, device='cpu')


    print("Running trained model for 3 episodes...")
    for episode in range(3):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use the trained model to predict the next action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")

    env.close()
    print("Evaluation finished.")
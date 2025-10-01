from stable_baselines3.common.env_checker import check_env
import os

from src.envs.env import BaseEnv, get_min_z

def check_custom_env(urdf_file):
    """
    Function to check the custom environment for compliance with Gymnasium standards.
    """
    # Create the environment instance
    min_z = get_min_z(urdf_file)
    env = BaseEnv(render_mode='headless', urdf_filename=urdf_file, start_position=[0, 0, -min_z])
    
    # Check the environment
    check_env(env, warn=True)
    
    print("Environment check complete!")

if __name__ == "__main__":
    # Example usage: replace with the desired URDF file path
    urdf_file = "robots/full_servobot/catbot.urdf"
    if not os.path.exists(urdf_file):
        print(f"Error: URDF file not found at {urdf_file}")
    else:
        check_custom_env(urdf_file)
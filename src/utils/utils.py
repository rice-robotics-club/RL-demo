
import os
from .config import ROBOTS
from . import config


def load_all_params(robot_name):
    ''' Load all parameters from config.py, using zero if not found.'''
    
    possible_params = [
        'GOAL_APPROACH_WEIGHT',
        'GOAL_REACHED_BONUS',
        'UPRIGHT_REWARD_WEIGHT',
        'ACTION_PENALTY_WEIGHT',
        'SHAKE_PENALTY_WEIGHT',
        'SURVIVAL_WEIGHT',
        'FALLEN_PENALTY',
        'FORWARD_VEL_WEIGHT',
        'JUMP_PENALTY_WEIGHT',
        'HIGH_ALTITUDE_PENALTY_WEIGHT',
        'HOME_POSITION_PENALTY_WEIGHT',
        'TILT_PENALTY_WEIGHT',
        'ORIENTATION_REWARD_WEIGHT'
    ]
    params = {}
    for param in possible_params:
        params[param] = ROBOTS[robot_name].get(param, 0.0)  # Default to 0.0 if not found
    OutputString = "Loaded Parameters: \n==========================\n"
    for key, value in params.items():
        OutputString += f"{key} = {value}, \n" 
    print(OutputString)
    return params

def select_robot(load_model=True):
    ''' Returns URDF file path, save path, save prefix for a given robot name. '''
    robot_options = list(ROBOTS.keys())
    print("Select Robot Name for Training (options:", ", ".join(robot_options), "): ")
    robot_name = input().strip()

    if robot_name in ROBOTS:

        
        
        print("Loading Robot Data...")
        print("==========================")
        print("Robot Selected: ", robot_name)

        if load_model:
            model_directory_list = os.listdir(ROBOTS[robot_name]['save_path'])
            print("Model Directory List: ", model_directory_list)
        
            # Handle case with no models found
            if len(model_directory_list) == 0:
                print("No saved models found. Please train your model first using train_to_objective.py.")
                exit(1)
            best_model_name = model_directory_list[-1]
            best_model_path = os.path.join(ROBOTS[robot_name]['save_path'], best_model_name)
            
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
                        # Check if the selected thing is a directory. If so, prompt for files within it. 
                        # This isn't recursive but I don't think we actually need a full recursive search function lol
                        selected_model = model_directory_list[selected_idx]
                        selected_model_path = os.path.join(ROBOTS[robot_name]['save_path'], selected_model)
                        if os.path.isdir(selected_model_path):
                            sub_files = os.listdir(selected_model_path)
                            print(f"'{selected_model}' is a directory. Available files:")
                            for sub_idx, sub_file in enumerate(sub_files):
                                print(f"{sub_idx + 1}: {sub_file}")
                            print("Enter the number of the file you want to use: ")
                            sub_selected_idx = int(input().strip()) - 1
                            if 0 <= sub_selected_idx < len(sub_files):
                                model_name = sub_files[sub_selected_idx]
                                model_path = os.path.join(ROBOTS[robot_name]['save_path'], selected_model, model_name)
                            else:
                                print("Invalid selection lmao. Exiting.")
                                exit(1)
                        else:
                            model_name = model_directory_list[selected_idx]
                            model_path = os.path.join(ROBOTS[robot_name]['save_path'], model_name)
                    else:
                        print("Invalid selection lmao. Exiting.")
                        exit(1)
                else:
                    # Just use the latest (assuming it's last in the directory) model
                    print("Using the latest model: [", best_model_name, "] by default!")
                    model_path = best_model_path
            print("Best Model Path: ", model_path)
        print("URDF File: ", ROBOTS[robot_name]['urdf_file'])
        print("Save Path: ", ROBOTS[robot_name]['save_path'])
        print("Robot Data Loaded.")
        print("==========================")
        if load_model:
            return ROBOTS[robot_name]['urdf_file'], ROBOTS[robot_name]['save_path'], ROBOTS[robot_name]['save_prefix'], model_path
        else:
            return ROBOTS[robot_name]['urdf_file'], ROBOTS[robot_name]['save_path'], ROBOTS[robot_name]['save_prefix']
    else:
        raise ValueError(f"Robot '{robot_name}' not found in configuration. If this isn't a typo, please update config.py to add it!")

def plot_moving_average(data, window_size=100):
    ''' Plots a moving average of the given data with the specified window size. '''
    import matplotlib.pyplot as plt
    import numpy as np

    if len(data) < window_size:
        print("Data length is less than window size. Cannot compute moving average.")
        return

    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    # ha ha
    moving_aves = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    plt.figure(figsize=(10, 5))
    plt.plot(moving_aves)
    plt.title(f'Moving Average (window size={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.show()
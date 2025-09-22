
import os
from .config import ROBOTS
from . import config

def load_param(name, default):
            return getattr(config, name, default)
         
def load_all_params():
    params = {
        "FORWARD_VEL_WEIGHT": load_param("FORWARD_VEL_WEIGHT", 1.0),
        "UPRIGHT_REWARD_WEIGHT": load_param("UPRIGHT_REWARD_WEIGHT", 0.5),
        "ACTION_PENALTY_WEIGHT": load_param("ACTION_PENALTY_WEIGHT", 0.01),
        "SHAKE_PENALTY_WEIGHT": load_param("SHAKE_PENALTY_WEIGHT", 0.01),
        "JUMP_PENALTY_WEIGHT": load_param("JUMP_PENALTY_WEIGHT", 5.0),
        "HIGH_ALTITUDE_PENALTY_WEIGHT": load_param("HIGH_ALTITUDE_PENALTY_WEIGHT", 2.0),
        "FALLEN_PENALTY": load_param("FALLEN_PENALTY", 20.0),
        "GOAL_APPROACH_WEIGHT": load_param("GOAL_APPROACH_WEIGHT", 2.0),
        "GOAL_REACHED_BONUS": load_param("GOAL_REACHED_BONUS", 100.0)
    }
    
    return params

def select_robot():
    ''' Returns URDF file path, save path, and save prefix for a given robot name. '''
    robot_options = list(ROBOTS.keys())
    print("Select Robot Name for Training (options:", ", ".join(robot_options), "): ")
    robot_name = input().strip()

    if robot_name in ROBOTS:
        model_directory_list = os.listdir(ROBOTS[robot_name]['save_path'])
        
        
        print("Loading Robot Data...")
        print("==========================")
        print("Robot Selected: ", robot_name)
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
        return ROBOTS[robot_name]['urdf_file'], ROBOTS[robot_name]['save_path'], ROBOTS[robot_name]['save_prefix'], model_path
    else:
        raise ValueError(f"Robot '{robot_name}' not found in configuration. If this isn't a typo, please update config.py to add it!")

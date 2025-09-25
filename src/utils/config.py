''' 
Parameters for training are stored here! 
'''


# --- REWARD WEIGHTS (TUNE THESE) ---
## All weights should be >= 0.0. Penalties are subtracted, not added. ## 

GOAL_APPROACH_WEIGHT = 5.0
GOAL_REACHED_BONUS = 200.0  # Large bonus on touching the goal box
UPRIGHT_REWARD_WEIGHT = 0.5
ACTION_PENALTY_WEIGHT = 0.001
SHAKE_PENALTY_WEIGHT = 0.001
SURVIVAL_BONUS = 0
FALLEN_PENALTY = 2.0
FORWARD_VEL_WEIGHT = 4.0
# New: discourage jumping/high vertical motion.
JUMP_PENALTY_WEIGHT = 0.1     # Penalize excessive vertical velocity
HIGH_ALTITUDE_PENALTY_WEIGHT = 0.1  # Penalize staying too high above ground



# Local filepaths for differerent robot configurations

ROBOTS = {
    'simple_quadruped': {
        'urdf_file': "robots/simple_quadruped.urdf",
        'save_path': 'models/quadruped_checkpoints/',
        'save_prefix': 'quadruped_model',
        
    },
    'servobot': {
        'urdf_file': "robots/servobot/servobot.urdf",
        'save_path': 'models/servobot_checkpoints/',
        'save_prefix': 'servobot_model'
    },
    'servobot_box': {
        'urdf_file': "robots/full_servobot/catbot.urdf",
        'save_path': 'models/servobot_checkpoints_box/',
        'save_prefix': 'servobot_box_model'
    }
}

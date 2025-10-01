''' 
Parameters for training are stored here! 
Also, local filepaths for different robot configurations.
All weights should be >= 0.0. Penalties are subtracted, not added.
'''

ROBOTS = {
    'simple_quadruped': {
        'urdf_file': "robots/simple_quadruped.urdf",
        'save_path': 'models/quadruped_checkpoints/',
        'save_prefix': 'quadruped_model',
        'start_position': [0, 0, 0.2],
         ### WEIGHTS FOR TRAINING ###
        #GOAL_APPROACH_WEIGHT': 5.0,
        #'GOAL_REACHED_BONUS': 200.0,  # Large bonus on touching the goal box
        'UPRIGHT_REWARD_WEIGHT': 0.5,
        'ACTION_PENALTY_WEIGHT': 0.01,
        'SHAKE_PENALTY_WEIGHT': 0.01,
        'SURVIVAL_BONUS': 0,
        'FALLEN_PENALTY': 2.0, 
        'FORWARD_VEL_WEIGHT': 100.0,
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1  # Penalize staying too high above ground """
    },


    'servobot': {
        'urdf_file': "robots/full_servobot/servobot.urdf",
        'save_path': 'models/servobot_checkpoints/',
        'save_prefix': 'servobot_model',
        'start_position': [0, 0, 0.2],
         ### WEIGHTS FOR TRAINING ###
        'GOAL_APPROACH_WEIGHT': 5.0,
        'GOAL_REACHED_BONUS': 200.0,  # Large bonus on touching the goal box
        'UPRIGHT_REWARD_WEIGHT': 0.5,
        'ACTION_PENALTY_WEIGHT': 0.01,
        'SHAKE_PENALTY_WEIGHT': 0.01,
        'SURVIVAL_BONUS': 0,
        'FALLEN_PENALTY': 2.0,
        'FORWARD_VEL_WEIGHT': 14.0,
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1  # Penalize staying too high above ground
    },

    
    'servobot_box': {
        'urdf_file': "robots/full_servobot/catbot.urdf",
        'save_path': 'models/servobot_checkpoints_box/',
        'save_prefix': 'servobot_box_model',
        'start_position': [0, 0, 0.2],
        ### WEIGHTS FOR TRAINING ###
        'GOAL_APPROACH_WEIGHT': 15.0,
        'GOAL_REACHED_BONUS': 200.0,  # Large bonus on touching the goal box
        'UPRIGHT_REWARD_WEIGHT': 0.5,
        'ACTION_PENALTY_WEIGHT': 0.1,
        'SHAKE_PENALTY_WEIGHT': 0.1,
        'SURVIVAL_BONUS': 0,
        'FALLEN_PENALTY': 2.0,
        'FORWARD_VEL_WEIGHT': 4.0,
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1  # Penalize staying too high above ground
    }
}

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
        'UPRIGHT_REWARD_WEIGHT': 0.75,
        'SHAKE_PENALTY_WEIGHT': 0.05,
        'SURVIVAL_WEIGHT': 0.001,  # Small constant reward for each step the robot stays upright
        'FALLEN_PENALTY': 20,
        'FORWARD_VEL_WEIGHT': 2.0,
        'ANGULAR_VEL_WEIGHT': 2.0,  # New: reward for turning (yaw angular velocity)
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1,  # Penalize staying too high above ground
        # New: Home Position Weight
        'HOME_POSITION_PENALTY_WEIGHT': .5,  # Penalize distance from home position
        'TILT_PENALTY_WEIGHT': 0.05,  # Penalize excessive tilting (pitch/roll)
        'ACTION_LIMIT': 0.2, # Proportional limit on joint angles. Should be between 0 and 1 # 0 is default and means no restriction. Otherwise smaller limit means more restriction.
        'INITIAL_MOMENTUM': 1.0,  # Scale of random initial momentum at start of episode (0.0 to 1.0'ACTION_LIMIT': 0.2 # Proportional limit on joint angles. Should be between 0 and 1 # 0 is default and means no restriction. Otherwise smaller limit means more restriction.
    },


    'servobot': {
        'urdf_file': "robots/full_servobot/servobot.urdf",
        'save_path': 'models/servobot_checkpoints/',
        'save_prefix': 'servobot_model',
        'start_position': [0, 0, 0.2],
        ### WEIGHTS FOR TRAINING ###
        'UPRIGHT_REWARD_WEIGHT': 0.75,
        'SHAKE_PENALTY_WEIGHT': 0.05,
        'SURVIVAL_WEIGHT': 0.001,  # Small constant reward for each step the robot stays upright
        'FALLEN_PENALTY': 20,
        'FORWARD_VEL_WEIGHT': 2.0,
        'ANGULAR_VEL_WEIGHT': 2.0,  # New: reward for turning (yaw angular velocity)
        # New: discourage jumping/high vertical motion.
        'JUMP_PENALTY_WEIGHT': 0.1,     # Penalize excessive vertical velocity
        'HIGH_ALTITUDE_PENALTY_WEIGHT': 0.1,  # Penalize staying too high above ground
        # New: Home Position Weight
        'HOME_POSITION_PENALTY_WEIGHT': .5,  # Penalize distance from home position
        'TILT_PENALTY_WEIGHT': 0.05,  # Penalize excessive tilting (pitch/roll)
        'ACTION_LIMIT': 0.2, # Proportional limit on joint angles. Should be between 0 and 1 # 0 is default and means no restriction. Otherwise smaller limit means more restriction.
        'INITIAL_MOMENTUM': 1.0,  # Scale of random initial momentum at start of episode (0.0 to 1.0
    },
}



# --- REWARD WEIGHTS (TUNE THESE) ---
GOAL_APPROACH_WEIGHT = 5.0
GOAL_REACHED_BONUS = 200.0  # Large bonus on touching the goal box
UPRIGHT_REWARD_WEIGHT = 0.5
ACTION_PENALTY_WEIGHT = 0.001
SHAKE_PENALTY_WEIGHT = 0.001
SURVIVAL_BONUS = 0
FALLEN_PENALTY = 2.0
FORWARD_VEL_WEIGHT = 3.0
# New: discourage jumping/high vertical motion.
JUMP_PENALTY_WEIGHT = 0.2     # Penalize excessive vertical velocity
HIGH_ALTITUDE_PENALTY_WEIGHT = 0.1  # Penalize staying too high above ground

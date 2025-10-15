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
import time
import math
import json
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from ..utils import utils

from ..utils import config

from ..utils.kinematics import IK


'''
Ideal Structure:
            ______________           _____________      __________________ 
                train.py  |         | visualize.py |    |    test.py     |
           ----------------          --------|------    ----------------
                     \ _________________     |     ____________/
                                        \ ______ /
                                        | env.py |
                                         --------
 
 '''

# --- Helpful functions ---
def get_min_z(urdf_path):
    """
    Loads a URDF and returns the smallest Z coordinate of its combined bounding box.
    """
    p.connect(p.DIRECT)
    # Load the URDF file
    try:
        robot_id = p.loadURDF(urdf_path)
    except p.error as e:
        p.disconnect()
        raise FileNotFoundError(f"Failed to load URDF: {urdf_path}. Error: {e}")

    # Initialize min_z to a very large value
    min_z = 10

    # The base of the model is treated as link -1
    # Then, iterate through all other links (joints)
    num_joints = p.getNumJoints(robot_id)
    link_indices = [-1] + list(range(num_joints))

    for link_index in link_indices:
        # getAABB returns (min_coords, max_coords)
        aabb = p.getAABB(robot_id, link_index)
        aabb_min = aabb[0]
        current_min_z = aabb_min[2] # Z is the third coordinate (index 2)

        # Update the overall minimum Z value if the current link is lower
        if current_min_z < min_z:
            min_z = current_min_z
    p.disconnect()
    return min_z


# --- Custom Gymnasium Environment for our Robots ---
class BaseEnv(gym.Env):
    """
    A custom environment that wraps the PyBullet simulation for
    reinforcement learning.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 240}

    # Change: init now accepts a target box (center and size).
    def __init__(self, 
                 render_mode=None, 
                 urdf_filename="simple_quadruped.urdf", 
                 start_position=[0, 0, 1],
                 target_speed = .5,):
        super(BaseEnv, self).__init__()
        '''
        This class implements the custom Gym environment for our robot RL training!
        '''

        self.urdf_filename = urdf_filename

        # Decide between PyBullet's GUI and Headless modes of operation
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            self.debug_line_id = None  # To store the line ID for rendering
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Environment constants
        self.time_step = 1.0 / 240.0
        self.episode_duration = 5.0  # Slightly longer to allow exploration
        self.steps_per_episode = int(self.episode_duration / self.time_step)
        self.action_force_limit = 50
        
        self.action_skip = 10

        params = utils.load_all_params(robot_name=os.path.splitext(os.path.basename(urdf_filename))[0])
        for param, value in params.items():
                setattr(self, param, value)
        # load parameters from config.py
        

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(bodyUniqueId=self.plane_id, 
                 linkIndex=-1,      # -1 for the base
                 lateralFriction=0.8)

        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.start_position = start_position
        self.robot_id = p.loadURDF(self.urdf_filename, self.start_position, start_orientation, useFixedBase=False,flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.start_position = base_pos
        self.joint_limit = 1.57
        self.action_factor = self.joint_limit
        if self.ACTION_LIMIT >0:
            self.action_factor *= self.ACTION_LIMIT
        self.initialize_joints()

        # Define some stable 'home' position array for each joint (straight down, angle of 0)
        self.home_position = [0 for _ in self.joint_indices]
        if 'servobot' in urdf_filename:
            ik = IK()
            idle_cfg = ik.get_idle_cfg(height=0.18)
            self.home_position = [0]*len(self.joint_indices)
            for i in self.joint_indices:
                self.home_position[i] = idle_cfg[self.get_joint_name[i]]
        self.previous_action = np.zeros(self.action_space.shape)

        # Generate a random target velocity to start (in the x-y plane, with a 0 component in the z direction)
        self.target_speed = target_speed
        self.target_velocity = self.generate_random_target_velocity(target_speed)

        # Generate a random target turn to start (yaw angle in radians between -pi and pi)
        self.target_turn = self.generate_random_turn_vector()

        # Generate an initial momentum vector to start (in the x-y plane, with a 0 component in the z direction)
        self.initial_momentum_vector = self.generate_random_initial_momentum(strength=0.0)

        self.render_mode = render_mode

    def initialize_joints(self):
        self.joint_indices = []
        self.get_joint_name = {}
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            self.get_joint_name[i] = joint_info[1].decode('utf-8')
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
            # restrict joint angles
            if len(joint_info) >= 9:
                self.joint_limit = min(self.joint_limit, joint_info[8],joint_info[9])
            # restrict action force limit based on joint max force if available
            if len(joint_info) >= 11:
                self.action_force_limit = min(self.action_force_limit, joint_info[10])
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_joints,), dtype=np.float32)

        # Define the size of our observation space based on several components:
        # 1. Joint positions and velocities (2 values per joint)
        # 2. Base position (3), 
        # 3. Orientation (4), 
        # 4. Linear velocity (3), 
        # 5. Angular velocity (3),
        # 6. Cosine and Sine of joint angles (2 values per joint)
        # 7. Control Goal Velocity (3 values: x,y,z components)
        # 8. Control Goal Turn (1 value, in positive or negative radians/sec)
        obs_space_shape = (num_joints * 4) + 13 + 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_shape,), dtype=np.float32)

    def _get_obs(self):
        '''
        Returns the agent's observation of the environment. 
        (This is basically the list of state variables the agent sees.)
        '''
        # Angles and velocities of all the joints
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        ## TESTING: ##
        # let's try adding in the cosine and sine values of these joint angles 
        # (This can help with angle wrapping issues)
        joint_cos = [math.cos(pos) for pos in joint_positions]
        joint_sin = [math.sin(pos) for pos in joint_positions]

        # Robot base (central body) 
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)
        
        # Get the goal velocity vector
        target_vel = self.target_velocity
        
        # Get the goal turn vector
        target_turn = self.target_turn

        # Compose the full observation vector and return it
        obs = np.concatenate([
            joint_positions, joint_velocities, joint_cos, joint_sin, base_pos, base_orient,
            base_vel, base_angular_vel, target_vel, [target_turn]
        ])
        return obs.astype(np.float32)

    def generate_random_target_velocity(self, target_speed):
        ''' Generates a random target velocity vector in the x-y plane with a 0 component in the z direction 
        and a maginitude between min_speed and max_speed '''
        angle = np.random.uniform(0, 2 * np.pi)
        # Have some variation in the target speed to make the policy more robust
        speed = np.random.uniform(target_speed - .25, target_speed + .25)
        return np.array([speed * np.cos(angle), speed * np.sin(angle), 0])

    def generate_random_turn_vector(self):
        ''' Generates a random turn command (yaw angle in radians/sec between -pi/2 and pi/2) '''
        theta = np.random.uniform(-np.pi/2, np.pi/2)
        return theta
    
    def generate_random_initial_momentum(self, strength):
        ''' Generates a random initial momentum vector in the x-y plane with a magnitude up to 'strength' '''
        angle = np.random.uniform(0, 2 * np.pi)
        momentum = np.random.uniform(0, strength)
        return np.array([momentum * np.cos(angle), momentum * np.sin(angle), 0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_position = self.start_position
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.robot_id, start_position, start_orientation)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])

        for joint_index in self.joint_indices:
            p.resetJointState(self.robot_id, joint_index, targetValue=0, targetVelocity=0)
            p.setJointMotorControl2(
                self.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=0, force=self.action_force_limit
            )

        self.steps_taken = 0


        # New target velocity for this episode
        self.target_velocity = self.generate_random_target_velocity(self.TARGET_SPEED)
        # New target orientation for this episode
        self.target_turn = self.generate_random_turn_vector()

        # New initial momentum for this episode
        self.initial_momentum_vector = self.generate_random_initial_momentum(strength=self.INITIAL_MOMENTUM)
        p.resetBaseVelocity(self.robot_id, linearVelocity=self.initial_momentum_vector.tolist(), angularVelocity=[0,0,0])

        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def calculate_step_reward_new(self, action, steps_taken=0):
        ''' 
        This function implements the reward function described in https://federicosarrocco.com/blog/Making-Quadrupeds-Learning-To-Walk
        '''

        # NEW WITH THIS VERSION: 
        # There should be some 'goal' velocity determined at the beginning of each episode to simulate control.
        # We should get that and use it to calculate a reward for movement in the target direction. 
        # There is no 'target box' in this setup - instead it's just trying to follow a velocity vector.
       
        # This target velocity should be fairly small (0.5 m/s?) to start with - 
        # perhaps we can increase the speed as the training progresses?

        # We also want to define a 'home' position for each joint (probably in the __init__ method) 
        # and punish actions that move too far away from it. This will keep the robot more stable.

        # Get position, orientation, velocity
        current_base_pos, current_base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)
        target_vel = self.target_velocity

        # velocity commands
        target_angular_vel = self.target_turn * np.array([0,0,1])  # Yaw only
        target_z = self.start_position[2]

        is_fallen = current_base_pos[2] < 0.05

        ## Reward Components: ##
        # 1. Linear Velocity Tracking Reward
        r_lin_vel = self.FORWARD_VEL_WEIGHT * np.exp(-np.linalg.norm(np.array(base_vel) - np.array(target_vel))**2)
        # 2. Angular Velocity Tracking Reward
        r_ang_vel = self.ANGULAR_VEL_WEIGHT * np.exp(-np.linalg.norm(np.array(base_angular_vel) - np.array(target_angular_vel))**2 )
        # 3. Height Penalty
        r_height = -(current_base_pos[2] - target_z)**2
        # 4. Pose Similarity Penalty
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = np.array([state[0] for state in joint_states])
        r_pose = -(np.linalg.norm(joint_positions - np.array(self.home_position))**2)
        # 5. Action Rate Penalty
        r_action_rate = -np.linalg.norm(action-self.previous_action)**2
        # 6. Vertical Velocity Penalty
        r_lin_vel_z = -base_vel[2]**2
        # 7. Roll and Pitch Penalty
        rot_matrix = p.getMatrixFromQuaternion(current_base_orient)
        z_direction = np.array([rot_matrix[6], rot_matrix[7], rot_matrix[8]])
        if not (0.99<np.linalg.norm(z_direction) < 1.01):
            raise ValueError("Z direction vector is not normalized!")
        r_rp = -(1 - z_direction[2])  # Not the same as the penalty described in the blog, but approximately the same when the robot is almost upright.
        # 8. Survival Reward
        r_survival = (self.SURVIVAL_WEIGHT * 1) if not is_fallen else 0.0
        # 9. Fallen Penalty
        r_fallen = -self.FALLEN_PENALTY if is_fallen else 0.0

        ## Calculate total reward:
        total_reward = (r_lin_vel+r_ang_vel+ r_height + r_pose + r_action_rate + r_lin_vel_z + r_rp + r_survival - r_fallen)
        return total_reward
    
    def calculate_step_reward(self, action, steps_taken=0):
        ''' 
        This function is run for each physics step to calculate the reward earned by the robot during that step.
        '''

        # NEW WITH THIS VERSION: 
        # There should be some 'goal' velocity determined at the beginning of each episode to simulate control.
        # We should get that and use it to calculate a reward for movement in the target direction. 
        # There is no 'target box' in this setup - instead it's just trying to follow a velocity vector.
       
        # This target velocity should be fairly small (0.5 m/s?) to start with - 
        # perhaps we can increase the speed as the training progresses?

        # We also want to define a 'home' position for each joint (probably in the __init__ method) 
        # and punish actions that move too far away from it. This will keep the robot more stable.

        # Get position, turn, velocity
        current_base_pos, current_base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)
        rot_matrix = p.getMatrixFromQuaternion(current_base_orient)
        local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        forward_vector = np.array([-rot_matrix[3],rot_matrix[0], rot_matrix[6]])

        # Get the target velocity and turn vectors
        target_vel = self.target_velocity
        target_turn = self.target_turn

        ## Reward Components: ##
        # - Velocity in target direction: get the component of the base velocity in the direction of the target velocity
        goal_component = np.dot(base_vel, target_vel) / (np.linalg.norm(target_vel) + 1e-6)
        # This should be positive if moving in the right direction, negative if moving away
        goal_velocity_reward = self.FORWARD_VEL_WEIGHT * goal_component

        # - Uprightness
        uprightness = local_up_vector[2]
        upright_reward = self.UPRIGHT_REWARD_WEIGHT * uprightness
        
        is_fallen = current_base_pos[2] < 0.1 or uprightness < 0.5
        # Survival reward that ramps up over time to encourage longer episodes
        survival_reward = self.SURVIVAL_WEIGHT * steps_taken if not is_fallen else 0.0
        
        # Matching orientation with the target velocity direction (encourage facing the direction of movement)
   

        target_direction = target_vel / (np.linalg.norm(target_vel) + 1e-6)
        orientation_alignment = np.dot(forward_vector, target_direction)
        orientation_reward = self.ORIENTATION_REWARD_WEIGHT * max(0.0, orientation_alignment)

        ## - Penalties: ##
        # - Shaking
        shake_penalty = self.SHAKE_PENALTY_WEIGHT * np.sum(np.square(base_angular_vel))
        # - Falling Penalty

        fallen_penalty = self.FALLEN_PENALTY if is_fallen else 0.0
        # - Distance from home position 
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = np.array([state[0] for state in joint_states])
        home_deviation = np.sum(np.square(joint_positions - np.array(self.home_position)))
        home_penalty = self.HOME_POSITION_PENALTY_WEIGHT * home_deviation
        # - Jumping
        jump_penalty = self.JUMP_PENALTY_WEIGHT * abs(base_vel[2])
        # - High Altitude
        high_alt_pen = self.HIGH_ALTITUDE_PENALTY_WEIGHT * max(0.0, current_base_pos[2]-1.0)
        # - Tilting
        tilt_penalty = self.TILT_PENALTY_WEIGHT * (1.0 - uprightness)

        # Sum all components, return total reward
        total_reward = (
            goal_velocity_reward + upright_reward + survival_reward + orientation_reward - home_penalty -
            shake_penalty - fallen_penalty - jump_penalty - high_alt_pen - tilt_penalty
        )
        
        if steps_taken % 240 == 0 and self.render_mode == 'human':
            print("================= Step Reward Breakdown ===============")
            print(f"Target Velocity: {target_vel}, Current Velocity: {base_vel}")
            print(f"Base Position: {current_base_pos}, Uprightness: {uprightness:.2f}")
            print(f"Is Fallen: {is_fallen}")

            print(f"Step Reward Breakdown: Forward: {goal_velocity_reward:.2f}, Upright: {upright_reward:.2f}, Survival: {survival_reward:.2f}, Home Penalty: {-home_penalty:.2f}, "
              f"Shake Penalty: {-shake_penalty:.2f}, Fallen Penalty: {-fallen_penalty:.2f}, Orientation: {orientation_reward:.2f}, Tilt Penalty: {-tilt_penalty:.2f}, "
              f"Jump Penalty: {-jump_penalty:.2f}, High Alt Penalty: {-high_alt_pen:.2f} => Total: {total_reward:.2f}")

        return total_reward
        # DEBUG: Print out all the reward components

    def update_config(self, new_config):
        '''
        Update environment parameters based on a new configuration dictionary.
        '''
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: {key} is not a valid parameter of the environment.")
    
    def step(self, action):
            """
            Take a step in the simulation with a revised reward function and a strict no-jump rule.
            """
            total_reward = 0.0

            # Repeat the action for some number of steps equal to our action_skip value (to simulate lower control frequency)
            for _ in range(self.action_skip):
                # Iterate over each joint and attempt to move it to the calculated target position given by the policy
                for i, joint_index in enumerate(self.joint_indices):
                    p.setJointMotorControl2(
                        self.robot_id, joint_index, p.POSITION_CONTROL,
                        targetPosition= self.action_factor*action[i]+self.home_position[i], force=self.action_force_limit
                    )
                p.stepSimulation()
                self.steps_taken += 1
                # NEW: use alternate function, input steps taken for ramping survival reward
                total_reward += self.calculate_step_reward_new(action, steps_taken=self.steps_taken)

                if self.steps_taken >= self.steps_per_episode:
                    break
                if self.render_mode == 'human':
                    time.sleep(self.time_step)

            
            # --- Termination conditions ---

            terminated = False
            truncated = self.steps_taken >= self.steps_per_episode  # Timeout => truncated 

            # --- ▼▼▼ CORRECTED LOGIC BLOCK ▼▼▼ ---

            # 1. Get BOTH final position and final orientation
            final_pos, final_orientation = p.getBasePositionAndOrientation(self.robot_id)

            # 2. Check for jumping
            if final_pos[2] > 1.3:
                print("🚫 Jump Detected! Episode terminated with penalty. 🚫")
                terminated = False

            # 3. Check for falling (using the correct orientation variable)
            rotation_matrix = p.getMatrixFromQuaternion(final_orientation)
            final_up_vector = np.array([rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]])
            if final_pos[2] < 0.1 or final_up_vector[2] < 0.3:
                terminated = True
                #print("🤖 Robot has fallen! Episode terminated. 🤖")
                # Display a message in the GUI if in GUI mode
                if self.render_mode == 'human':
                    self.fallen_id = p.addUserDebugText("FALLEN!", [0,0,1], textColorRGB=[1,0,0], textSize=2.5, lifeTime=.1)
                    
            # --- ▲▲▲ END OF CORRECTION ▲▲▲ ---
            self.previous_action = action
            info = self._get_info()

            return self._get_obs(), total_reward, terminated, truncated, info

    def _get_info(self):
        '''
        Returns additional diagnostic information about the environment.
        '''
        info = {}
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)

        rot_matrix = p.getMatrixFromQuaternion(base_orient)
        local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        uprightness = local_up_vector[2]

        info['base_position'] = base_pos
        info['base_orientation'] = base_orient
        info['base_velocity'] = base_vel
        info['base_angular_velocity'] = base_angular_vel
        info['uprightness'] = uprightness

        return info
    
    def render(self):
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    urdf_file, save_path, save_prefix, model_path = utils.select_robot()

    # Pass box parameters into the environment.
    env = BaseEnv(
        render_mode='human', 
        urdf_filename=urdf_file, 
    )
    
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)  # Slightly larger n_steps may help with harder tasks

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix=save_prefix
    )
    
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)  # This task may require longer training
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")

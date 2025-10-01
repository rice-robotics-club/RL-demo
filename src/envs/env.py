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
                 target_box_center=[10.0, 0.0], 
                 target_box_size=[1.0, 1.0, 1.0]):
        super(BaseEnv, self).__init__()
        '''
        This class implements the custom Gym environment for our robot RL training!
        '''

        self.urdf_filename = urdf_filename

        # Decide between PyBullet's GUI and Headless modes of operation
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Environment constants
        self.time_step = 1.0 / 240.0
        self.episode_duration = 3.0  # Slightly longer to allow exploration
        self.steps_per_episode = int(self.episode_duration / self.time_step)
        self.action_force_limit = 20
        
        self.action_skip = 2 
        # Note: this was previously too high, leading to the robot only being able to make one or two moves before falling over.
        # 2-5 seems like a reasonable constraint. 

        params = utils.load_all_params(robot_name=os.path.splitext(os.path.basename(urdf_filename))[0])
        for param, value in params.items():
                setattr(self, param, value)
        # load parameters from config.py
        

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)
        self.plane_id = p.loadURDF("plane.urdf")
        

        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.start_position = start_position
        self.robot_id = p.loadURDF(self.urdf_filename, self.start_position, start_orientation, useFixedBase=False)
        
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.start_position = base_pos
        # New: create the target box.
        self.target_box_center = np.array(target_box_center, dtype=np.float32)
        self.target_box_size = np.array(target_box_size, dtype=np.float32)
        box_half_extents = self.target_box_size / 2.0
        box_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[0, 1, 0, 0.8])
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        self.box_id = p.createMultiBody(
            baseMass=0,  # Static body (immovable)
            baseCollisionShapeIndex=box_collision_shape_id,
            baseVisualShapeIndex=box_visual_shape_id,
            basePosition=[self.target_box_center[0], self.target_box_center[1], box_half_extents[2]]
        )

        self.last_distance_to_target = 0.0
        
        self.initialize_joints()
            
        self.render_mode = render_mode

    def initialize_joints(self):
            self.joint_indices = []
            num_joints = p.getNumJoints(self.robot_id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    self.joint_indices.append(i)
            self.action_space = spaces.Box(low=-1, high=1, shape=(num_joints,), dtype=np.float32)

            # Define the size of our observation space based on several components:
            # 1. Joint positions and velocities (2 values per joint)
            # 2. Base position (3), 
            # 3. Orientation (4), 
            # 4. Linear velocity (3), 
            # 5. Angular velocity (3),
            # 6. Cosine and Sine of joint angles (2 values per joint)
            # 7. Vector to target box center (2 values: x and y)
            obs_space_shape = (num_joints * 4) + 13 + 2
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
        
        # Change: observe the vector to the center of the target box.
        vec_to_target = self.target_box_center - np.array(base_pos[:2])

        # Compose the full observation vector and return it
        obs = np.concatenate([
            joint_positions, joint_velocities, joint_cos, joint_sin, base_pos, base_orient,
            base_vel, base_angular_vel, vec_to_target
        ])
        return obs.astype(np.float32)

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
        
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.last_distance_to_target = np.linalg.norm(self.target_box_center - np.array(base_pos[:2]))
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info


    def calculate_step_reward_alternate(self, action):
        pass

    def calculate_step_reward(self, action):
        ''' 
        This function is run for each physics step to calculate the reward earned by the robot during that step.
        '''

        # Get current position and orientation
        current_base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)


        # --- Reward function terms ---
        
        # Approach reward: reward for reducing distance to our box target (centered at target_box_center)
        current_distance_to_target = np.linalg.norm(self.target_box_center - np.array(current_base_pos[:2]))
        # Calculate the change in distance to the target
        distance_gained = self.last_distance_to_target - current_distance_to_target
        # Calculate the approach reward
        approach_reward = self.GOAL_APPROACH_WEIGHT * distance_gained
        to_target = self.target_box_center - np.array(current_base_pos[:2])
        dist = np.linalg.norm(to_target) + 1e-6
        dir_unit = to_target / dist

        # Instantaneous speed in the direction of the target (clamped to >= 0).
        # Simple forward speed: use the robot's base x velocity
        forward_speed = max(base_vel[0], 0.0)

        # Calculate the reward for 'forwards movement' towards the target
        forward_reward = self.FORWARD_VEL_WEIGHT * forward_speed
        self.last_distance_to_target = current_distance_to_target

        # Upright reward: reward for keeping the robot upright (based on local up vector's Z component)
        rot_matrix = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])
        local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        uprightness = local_up_vector[2]

        ### Penalties ###
        # Action penalty: small penalty for large actions (to encourage smoother motions)
        action_penalty = self.ACTION_PENALTY_WEIGHT * np.sum(np.square(action))
        # Shake penalty: small penalty for large angular velocities (to encourage stability)
        shake_penalty = self.SHAKE_PENALTY_WEIGHT * np.sum(np.square(base_angular_vel))
        
        # We have a huge penalty for falling over. This is a simple check to see if it's done that.
        # We check if the robot's base is too low or if it's tilted too far over.
        is_fallen = current_base_pos[2] < 0.6 or uprightness < 0.5
        fallen_penalty = self.FALLEN_PENALTY if is_fallen else 0.0
        
        step_reward = 0
        upright_reward = self.UPRIGHT_REWARD_WEIGHT * uprightness
        jump_penalty = self.JUMP_PENALTY_WEIGHT * abs(base_vel[2])
        high_alt_pen = self.HIGH_ALTITUDE_PENALTY_WEIGHT * max(0.0, current_base_pos[2]-1.0)
        step_reward -= (jump_penalty + high_alt_pen)
        step_reward = (
            approach_reward + forward_reward + upright_reward -
            action_penalty - shake_penalty - fallen_penalty
        )
        
        # DEBUG: Print out all the reward components
        #print(f"Step Reward Breakdown: Approach: {approach_reward:.2f}, Forward: {forward_reward:.2f}, Upright: {upright_reward:.2f}, "
        #      f"Action Penalty: {action_penalty:.2f}, Shake Penalty: {shake_penalty:.2f}, Fallen Penalty: {fallen_penalty:.2f}, "
        #      f"Jump Penalty: {jump_penalty:.2f}, High Alt Penalty: {high_alt_pen:.2f} => Total: {step_reward:.2f}")

        return step_reward
    
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
                        targetPosition= 1.57 * action[i], force=self.action_force_limit
                    )
                p.stepSimulation()
                self.steps_taken += 1

                total_reward += self.calculate_step_reward(action)

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
                total_reward -= 50.0
                terminated = False

            # 3. Check for falling (using the correct orientation variable)
            rotation_matrix = p.getMatrixFromQuaternion(final_orientation)
            final_up_vector = np.array([rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]])
            if final_pos[2] < 0.6 or final_up_vector[2] < 0.7:
                terminated = False

            # --- ▲▲▲ END OF CORRECTION ▲▲▲ ---

            # 4. Check for success
            contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.box_id)
            if len(contact_points) > 0 and not truncated:
                total_reward += self.GOAL_REACHED_BONUS
                truncated = True
                print("🎉🎉🎉 Goal Touched! 🎉🎉🎉")

            info = self._get_info()

            return self._get_obs(), total_reward, terminated, truncated, info

    def _get_info(self):
        '''
        Returns additional diagnostic information about the environment.
        '''
        info = {}
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)
        to_target = self.target_box_center - np.array(base_pos[:2])
        dist_to_target = np.linalg.norm(to_target)

        rot_matrix = p.getMatrixFromQuaternion(base_orient)
        local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        uprightness = local_up_vector[2]

        info['base_position'] = base_pos
        info['base_orientation'] = base_orient
        info['base_velocity'] = base_vel
        info['base_angular_velocity'] = base_angular_vel
        info['distance_to_target'] = dist_to_target
        info['uprightness'] = uprightness

        return info
    
    def render(self):
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    urdf_file, save_path, save_prefix, model_path = utils.select_robot()

    # Set target box center [x, y] and size [width, depth, height].
    box_center = [12.0, 3.0]
    box_size = [2.0, 2.0, 1.0]  # A 2x2x1 m box

    # Pass box parameters into the environment.
    env = BaseEnv(
        render_mode='human', 
        urdf_filename=urdf_file, 
        target_box_center=box_center,
        target_box_size=box_size
    )
    
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048)  # Slightly larger n_steps may help with harder tasks

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix=save_prefix
    )
    
    print(f"Starting training... Target Box Center: {box_center}, Size: {box_size}")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)  # This task may require longer training
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")

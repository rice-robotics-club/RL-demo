'''
Universal environment for training, testing, simulating, and running quadruped robots in PyBullet using OpenAI Gym interface.

'''

import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# --- Custom Gymnasium Environment for the Quadruped ---
class QuadrupedEnv(gym.Env):
    """
    A custom environment that wraps the PyBullet simulation for
    reinforcement learning.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 240}

    def __init__(self, render_mode=None, urdf_filename="simple_quadruped.urdf"):
        super(QuadrupedEnv, self).__init__()
        self.urdf_filename = urdf_filename

        # Connect to the PyBullet physics server
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Environment constants
        self.time_step = 1.0 / 240.0
        self.episode_duration = 2.0  # 10 seconds
        self.steps_per_episode = int(self.episode_duration / self.time_step)
        self.action_force_limit = 1 # Maximum force in Nm
        self.action_skip = 8 # Agent makes a decision every 8 simulation steps (30Hz)

        # --- REWARD WEIGHTS (TUNE THESE) ---
        self.FORWARD_VELOCITY_WEIGHT = -1.5
        self.UPRIGHT_REWARD_WEIGHT = 1.0
        self.ACTION_PENALTY_WEIGHT = 1.0
        self.SHAKE_PENALTY_WEIGHT = 0.1
        self.SURVIVAL_BONUS = 0.1
        self.FALLEN_PENALTY = 5.0

        # --- START POSITION AND ORIENTATION---
        self.start_position = [1.5, 1.5, 0.2]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # Set up the simulation environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # --- MODIFIED: Load robot and define spaces in __init__ ---
        start_position = self.start_position
        start_orientation = self.start_orientation
        self.robot_id = p.loadURDF(self.urdf_filename, start_position, start_orientation, useFixedBase=False)
        
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

        num_joints = len(self.joint_indices)
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(num_joints,), dtype=np.float32)

        obs_space_shape = (num_joints * 2) + 13
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_shape,), dtype=np.float32)
        # -----------------------------------------------------------
        
        self.render_mode = render_mode

    def _get_obs(self):
        """
        Get the current observation of the environment.
        """
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)
        
        obs = np.concatenate([
            joint_positions,
            joint_velocities,
            base_pos,
            base_orient,
            base_vel,
            base_angular_vel
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the simulation to its initial state.
        """
        super().reset(seed=seed)
        
        # --- MODIFIED: Reset robot state without reloading ---
        start_position = self.start_position
        start_orientation = self.start_orientation
        p.resetBasePositionAndOrientation(self.robot_id, start_position, start_orientation)
        p.resetBaseVelocity(self.robot_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])

        for joint_index in self.joint_indices:
            # Reset joint to position 0 with 0 velocity
            p.resetJointState(self.robot_id, joint_index, targetValue=0, targetVelocity=0)
            # Re-apply motor control
            p.setJointMotorControl2(
                self.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=0, force=self.action_force_limit
            )
        # ----------------------------------------------------

        self.steps_taken = 0
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """
        Take a step in the simulation with a revised reward function and action skipping.
        """
        total_reward = 0.0
        
        for _ in range(self.action_skip):
            # Apply the SAME action for the duration of the skip
            for i, joint_index in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=action[i],
                    force=self.action_force_limit
                )

            # Step the simulation forward
            p.stepSimulation()
            self.steps_taken += 1

            # --- GET STATE FOR REWARD CALCULATION ---
            current_base_pos, current_base_orient = p.getBasePositionAndOrientation(self.robot_id)
            base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)

            # --- Calculate individual reward/penalty components ---
            forward_velocity = base_vel[0]
            rot_matrix = p.getMatrixFromQuaternion(current_base_orient)
            local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
            uprightness = local_up_vector[2]
            
            action_penalty = self.ACTION_PENALTY_WEIGHT * np.sum(np.square(action))
            shake_penalty = self.SHAKE_PENALTY_WEIGHT * np.sum(np.square(base_angular_vel))

            # --- STATE-DEPENDENT REWARD LOGIC ---
            is_fallen = current_base_pos[2] < 0.6 or uprightness < 0.25

            
            step_reward = 0
            if not is_fallen:
                forward_reward = self.FORWARD_VELOCITY_WEIGHT * forward_velocity
                upright_reward = self.UPRIGHT_REWARD_WEIGHT * uprightness
                step_reward = (
                    forward_reward + 
                    upright_reward + 
                    self.SURVIVAL_BONUS - 
                    action_penalty - 
                    shake_penalty
                )
            else:
                upright_reward = self.UPRIGHT_REWARD_WEIGHT * uprightness
                step_reward = (
                    upright_reward - 
                    self.FALLEN_PENALTY - 
                    action_penalty - 
                    shake_penalty
                )

                # step_reward = -1000  # Large negative reward for falling
                # # End the episode immediately if fallen
                # self.steps_taken = self.steps_per_episode
            
            ''' REALLY SHITTY DEBUG INFO YOU CAN TURN ON IF YOU'RE CURIOUS (VISIBLE IN THE GUI) '''
            if self.render_mode == 'human':
                p.addUserDebugText(f"Step Reward: {step_reward:.4f}", [0,0,1.2], textColorRGB=[1,0,0], lifeTime=0.1)
            ''' ----------------------------------------------------- '''

            total_reward += step_reward

            if self.render_mode == 'human':
                time.sleep(self.time_step)

            # Check for termination inside the loop in case the episode ends mid-skip
            if self.steps_taken >= self.steps_per_episode:
                break
        
        terminated = self.steps_taken >= self.steps_per_episode
        truncated = False 
        info = {}

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        # We handle rendering directly in the PyBullet GUI.
        pass

    def close(self):
        p.disconnect()

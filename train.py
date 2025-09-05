# This script sets up a reinforcement learning environment to train a simple quadruped robot
# to move as far as possible in 10 seconds. It uses the PyBullet physics engine for the
# simulation and the stable-baselines3 library for the RL agent.
#
# To run this script, you'll need to install the required libraries:
# pip install pybullet gymnasium stable-baselines3[extra]
#
# The script now assumes that 'simple_quadruped.urdf' exists in the same directory.

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
        self.episode_duration = 10.0  # 10 seconds
        self.steps_per_episode = int(self.episode_duration / self.time_step)
        self.action_force_limit = 50.0 # Maximum force in Nm
        self.action_skip = 8 # Agent makes a decision every 8 simulation steps (30Hz)

        # --- REWARD WEIGHTS (TUNE THESE) ---
        self.FORWARD_VELOCITY_WEIGHT = 1.5
        self.UPRIGHT_REWARD_WEIGHT = 0.5
        self.ACTION_PENALTY_WEIGHT = 0.001
        self.SHAKE_PENALTY_WEIGHT = 0.001
        self.SURVIVAL_BONUS = 0.1
        self.FALLEN_PENALTY = 2.0

        # Set up the simulation environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # --- MODIFIED: Load robot and define spaces in __init__ ---
        start_position = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
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
        start_position = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
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
            is_fallen = current_base_pos[2] < 0.6 or uprightness < 0.75
            
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
            
            total_reward += step_reward

            # Check for termination inside the loop in case the episode ends mid-skip
            if self.steps_taken >= self.steps_per_episode:
                break
        
        terminated = self.steps_taken >= self.steps_per_episode
        truncated = False 
        info = {}
        
        if self.render_mode == 'human':
            time.sleep(self.time_step * self.action_skip)

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        # We handle rendering directly in the PyBullet GUI.
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    # To use a different robot, change the filename here
    urdf_file = "servobot/servobot.urdf"
    # Create the environment. Stable-baselines will automatically call reset.
    env = QuadrupedEnv(render_mode='human', urdf_filename=urdf_file)
    
    # Define the PPO agent from stable-baselines3
    model = PPO("MlpPolicy", env, verbose=1,n_steps=1024)

    # Setup Checkpoint Callback to save the model every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./quadruped_checkpoints/',
        name_prefix='quadruped_model'
    )
    
    # Train the agent for a number of timesteps
    print("Starting training...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")

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

    def __init__(self, render_mode=None):
        super(QuadrupedEnv, self).__init__()

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
        # --- NEW: Action skipping for more realistic control frequency ---
        self.action_skip = 8 # Agent makes a decision every 8 simulation steps (30Hz)
        # ----------------------------------------------------------------

        # --- REWARD WEIGHTS (TUNE THESE) ---
        # These weights control the importance of each reward component.
        self.FORWARD_VELOCITY_WEIGHT = 1.5
        self.UPRIGHT_REWARD_WEIGHT = 0.5
        self.ACTION_PENALTY_WEIGHT = 0.001
        self.SHAKE_PENALTY_WEIGHT = 0.001
        self.SURVIVAL_BONUS = 0.1
        # --- NEW: Add a penalty for being in a fallen state ---
        self.FALLEN_PENALTY = 2.0
        # ----------------------------------------------------

        # Set up the simulation environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        # Load the robot and ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Action space: target joint angles for the four revolute joints
        num_joints = 4
        # We can control the target position of each joint.
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(num_joints,), dtype=np.float32)

        # Observation space: joint angles, joint velocities, base position and orientation
        # 4 joint positions, 4 joint velocities, 3 base pos, 4 base orient, 3 base vel, 3 base angular vel
        # Total: 4 + 4 + 3 + 4 + 3 + 3 = 21
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_joints * 2 + 13,), dtype=np.float32)
        
        self.robot_id = None
        self.joint_indices = []
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
        
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        
        start_position = [0, 0, 1.0] # Starts at z=1.0 as requested
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Load the URDF file from the working directory
        self.robot_id = p.loadURDF("simple_quadruped.urdf", start_position, start_orientation, useFixedBase=False)
        
        # Find the revolute joints
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                p.setJointMotorControl2(
                    self.robot_id, i, p.POSITION_CONTROL, targetPosition=0, force=self.action_force_limit
                )

        self.steps_taken = 0
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """
        Take a step in the simulation with a revised reward function and action skipping.
        """
        total_reward = 0.0
        
        # --- MODIFIED: Loop for action skipping ---
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
        
        # --- MODIFIED: Use total accumulated reward ---
        terminated = self.steps_taken >= self.steps_per_episode
        truncated = False 
        info = {}
        
        if self.render_mode == 'human':
            # This sleep should now be for the duration of the action skip
            time.sleep(self.time_step * self.action_skip)

        # The observation is from the FINAL state after all skipped steps
        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        # We handle rendering directly in the PyBullet GUI.
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    # Create the environment with a human-readable GUI
    env = QuadrupedEnv(render_mode='human')
    
    # Define the PPO agent from stable-baselines3
    # MlpPolicy is a multi-layer perceptron neural network
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


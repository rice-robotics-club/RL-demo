# This script loads a trained Stable Baselines3 model and runs it in the PyBullet
# simulation for evaluation.
#
# To use this script, make sure you have a trained model checkpoint file in
# the './quadruped_checkpoints/' directory.

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# --- Custom Gymnasium Environment for the Quadruped ---
# (The environment must be identical to the one used for training)
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
        self.episode_duration = 5.0
        self.steps_per_episode = int(self.episode_duration / self.time_step)
        self.action_force_limit = 20.0

        # Set up the simulation environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)

        # Load the robot and ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # copied this logic from train.py
        start_position = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # Action space: target joint angles for the four revolute joints
        num_joints = 4
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(num_joints,), dtype=np.float32)

        # Observation space: joint angles, joint velocities, base position and orientation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_joints * 2 + 13,), dtype=np.float32)
        
        self.robot_id = p.loadURDF(self.urdf_filename, start_position, start_orientation, useFixedBase=False)
        self.joint_indices = []
        self.render_mode = render_mode

    def _get_obs(self):
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
        super().reset(seed=seed)
        
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        
        start_position = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot_id = p.loadURDF(self.urdf_filename, start_position, start_orientation, useFixedBase=False)
        
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                p.setJointMotorControl2(
                    self.robot_id, i, p.POSITION_CONTROL, targetPosition=0, force=self.action_force_limit
                )

        self.steps_taken = 0
        self.initial_base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=self.action_force_limit
            )

        p.stepSimulation()
        
        current_base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        distance_traveled = current_base_pos[0] - self.initial_base_pos[0]
        reward = distance_traveled
        
        self.steps_taken += 1
        
        terminated = self.steps_taken >= self.steps_per_episode
        truncated = False
        info = {}
        
        if self.render_mode == 'human':
            p.stepSimulation()
            time.sleep(self.time_step)

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    # Path to the saved model file
    #model_path = "./quadruped_checkpoints/v3/quadruped_model_200000_steps.zip"
    model_path = "./servobot_checkpoints/servobot_model_500000_steps.zip"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train your model first using train_quadruped.py.")
    else:
        # Create the environment with a human-readable GUI
        env = QuadrupedEnv(render_mode='human', urdf_filename="servobot/servobot.urdf")
        
        # Load the trained model
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=env, device='cpu')
        
        print("Running trained model for 3 episodes...")
        for episode in range(3):
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Use the trained model to predict the next action
                action, _states = model.predict(obs, deterministic=True)
                
                # Take the action in the environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
            print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")

        env.close()
        print("Evaluation finished.")
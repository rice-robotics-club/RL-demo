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

    # <-- 修改: 初始化参数变为目标方块的中心和大小
    def __init__(self, render_mode=None, urdf_filename="simple_quadruped.urdf", 
                 target_box_center=[10.0, 0.0], target_box_size=[1.0, 1.0, 1.0]):
        super(QuadrupedEnv, self).__init__()
        self.urdf_filename = urdf_filename

        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Environment constants
        self.time_step = 1.0 / 240.0
        self.episode_duration = 15.0  # <-- 稍微延长一点时间给机器人探索
        self.steps_per_episode = int(self.episode_duration / self.time_step)
        self.action_force_limit = 200.0
        self.action_skip = 4

        # --- REWARD WEIGHTS (TUNE THESE) ---
        self.GOAL_APPROACH_WEIGHT = 5.0
        self.GOAL_REACHED_BONUS = 200.0  # 触碰成功给予巨大奖励
        self.UPRIGHT_REWARD_WEIGHT = 0.5
        self.ACTION_PENALTY_WEIGHT = 0.001
        self.SHAKE_PENALTY_WEIGHT = 0.001
        self.SURVIVAL_BONUS = 0
        self.FALLEN_PENALTY = 2.0
        self.FORWARD_VEL_WEIGHT = 3.0  
        # <-- 新增: "禁止跳跃" 的惩罚权重
        self.JUMP_PENALTY_WEIGHT = 0.02  # 惩罚过大的垂直速度
        self.HIGH_ALTITUDE_PENALTY_WEIGHT = 0.01 # 惩罚离地过高

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step)
        self.plane_id = p.loadURDF("plane.urdf")
        
        start_position = [0, 0, 1.0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.urdf_filename, start_position, start_orientation, useFixedBase=False)

        # <-- 新增: 创建目标方块
        self.target_box_center = np.array(target_box_center, dtype=np.float32)
        self.target_box_size = np.array(target_box_size, dtype=np.float32)
        box_half_extents = self.target_box_size / 2.0
        box_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=[0, 1, 0, 0.8])
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        self.box_id = p.createMultiBody(
            baseMass=0,  # 静态物体，不会被推动
            baseCollisionShapeIndex=box_collision_shape_id,
            baseVisualShapeIndex=box_visual_shape_id,
            basePosition=[self.target_box_center[0], self.target_box_center[1], box_half_extents[2]]
        )

        self.last_distance_to_target = 0.0
        
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

        num_joints = len(self.joint_indices)
        self.action_space = spaces.Box(low=-1.57, high=1.57, shape=(num_joints,), dtype=np.float32)

        obs_space_shape = (num_joints * 2) + 13 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_shape,), dtype=np.float32)
        
        self.render_mode = render_mode

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)
        
        # <-- 修改: 观测目标变为方块的中心
        vec_to_target = self.target_box_center - np.array(base_pos[:2])

        obs = np.concatenate([
            joint_positions, joint_velocities, base_pos, base_orient,
            base_vel, base_angular_vel, vec_to_target
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        start_position = [0, 0, 1.0]
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
        info = {}
        return observation, info

    def step(self, action):
            """
            Take a step in the simulation with a revised reward function and a strict no-jump rule.
            """
            total_reward = 0.0
            
            for _ in range(self.action_skip):
                for i, joint_index in enumerate(self.joint_indices):
                    p.setJointMotorControl2(
                        self.robot_id, joint_index, p.POSITION_CONTROL,
                        targetPosition=action[i], force=self.action_force_limit
                    )
                p.stepSimulation()
                self.steps_taken += 1

                current_base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                base_vel, base_angular_vel = p.getBaseVelocity(self.robot_id)

                # --- Reward function part ---
                current_distance_to_target = np.linalg.norm(self.target_box_center - np.array(current_base_pos[:2]))
                distance_gained = self.last_distance_to_target - current_distance_to_target
                approach_reward = self.GOAL_APPROACH_WEIGHT * distance_gained
                to_target = self.target_box_center - np.array(current_base_pos[:2])
                dist = np.linalg.norm(to_target) + 1e-6
                dir_unit = to_target / dist
                # 机器人在该方向的瞬时线速度（只取正向）
                forward_speed = float(np.dot(np.array(base_vel[:2]), dir_unit))
                forward_speed = max(forward_speed, 0.0)

                forward_reward = self.FORWARD_VEL_WEIGHT * forward_speed
                self.last_distance_to_target = current_distance_to_target

                rot_matrix = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])
                local_up_vector = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
                uprightness = local_up_vector[2]
                action_penalty = self.ACTION_PENALTY_WEIGHT * np.sum(np.square(action))
                shake_penalty = self.SHAKE_PENALTY_WEIGHT * np.sum(np.square(base_angular_vel))
                
                is_fallen = current_base_pos[2] < 0.6 or uprightness < 0.75
                
                step_reward = 0
                if not is_fallen:
                    upright_reward = self.UPRIGHT_REWARD_WEIGHT * uprightness
                    jump_penalty = self.JUMP_PENALTY_WEIGHT * abs(base_vel[2])
                    high_alt_pen = self.HIGH_ALTITUDE_PENALTY_WEIGHT * max(0.0, current_base_pos[2]-1.0)
                    step_reward -= (jump_penalty + high_alt_pen)
                    step_reward = (
                        approach_reward + forward_reward -
                        action_penalty - shake_penalty
                    )
                else:
                    step_reward = -self.FALLEN_PENALTY 
                    total_reward += step_reward
                    terminated = True        # ← 加这一行
                    break 

                total_reward += step_reward

                if self.steps_taken >= self.steps_per_episode:
                    break
            
            # --- Termination conditions ---

            terminated = False
            truncated = self.steps_taken >= self.steps_per_episode  # 超时 => 截断 

            # --- ▼▼▼ CORRECTED LOGIC BLOCK ▼▼▼ ---

            # 1. Get BOTH final position and final orientation
            final_pos, final_orientation = p.getBasePositionAndOrientation(self.robot_id)

            # 2. Check for jumping
            if final_pos[2] > 1.3:
                print("🚫 Jump Detected! Episode terminated with penalty. 🚫")
                total_reward -= 50.0
                terminated = True

            # 3. Check for falling (using the correct orientation variable)
            rotation_matrix = p.getMatrixFromQuaternion(final_orientation)
            final_up_vector = np.array([rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]])
            if final_pos[2] < 0.6 or final_up_vector[2] < 0.7:
                terminated = True

            # --- ▲▲▲ END OF CORRECTION ▲▲▲ ---

            # 4. Check for success
            contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.box_id)
            if len(contact_points) > 0 and not truncated:
                total_reward += self.GOAL_REACHED_BONUS
                truncated = True
                print("🎉🎉🎉 Goal Touched! 🎉🎉🎉")

            info = {}
            
            if self.render_mode == 'human':
                time.sleep(self.time_step * self.action_skip)

            return self._get_obs(), total_reward, terminated, truncated, info


    def render(self):
        pass

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    urdf_file = "simple_quadruped.urdf"

    # 在这里设置目标方块的中心 [x, y] 和 大小 [width, depth, height]
    box_center = [12.0, 3.0]
    box_size = [2.0, 2.0, 1.0] # 一个 2x2x1 米的方块

    # 将方块信息传递给环境
    env = QuadrupedEnv(
        render_mode='human', 
        urdf_filename=urdf_file, 
        target_box_center=box_center,
        target_box_size=box_size
    )
    
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048) # 稍微增加 n_steps 可能有助于学习更复杂的任务

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./servobot_checkpoints_box/',
        name_prefix='servobot_model_box'
    )
    
    print(f"Starting training... Target Box Center: {box_center}, Size: {box_size}")
    try:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback) # 这种任务可能需要更长的训练时间
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        env.close()
    
    print("Training finished.")
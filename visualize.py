
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import numpy as np
import pybullet as p

from src.envs.env import BaseEnv, get_min_z
from src.utils import utils
from gymnasium import wrappers

'''
This script is a mismash of the quadreped run_trained.py and the servobot train.py scripts to load and run a trained servobot model
to demonstrate its movement in the pybullet gui. right now it looks pretty silly bc i just ran the training on my laptop for 15 minutes
but we could prolly get some better results if we ran it longer. 
'''

class VisualizationEnv(BaseEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    def __init__(self, urdf_filename, start_position=[0,0,0], target_speed=0.5, render_mode='human'):
        super().__init__(render_mode=render_mode, urdf_filename=urdf_filename, start_position=start_position, target_speed=target_speed)
        
        # Set up a interactive debug variable for pybullet to control: 
        #   - target velocity direction (0 to 1 times 2pi)
        #   - target velocity magnitude (0 to 1)
        #   - Target orientation (0 to 1 times 2pi)
        self.target_velocity_direction_id = p.addUserDebugParameter("Target Velocity Direction", 0, 1, 0)
        self.target_velocity_magnitude_id = p.addUserDebugParameter("Target Velocity Magnitude", 0, 1, 0)
        self.target_orientation_id = p.addUserDebugParameter("Target Orientation", 0, 1, 0)

        # Initialize debug object lines to be drawn on for visualization of orientation/velocity
        self.debug_lines = []

    def reset(self, seed=None, options=None):
        """Override reset to use slider values instead of random targets"""
        # Call parent reset first
        obs, info = super().reset(seed=seed, options=options)
        
        # Immediately override the random targets with slider values
        # Read the current slider positions
        try:
            direction = p.readUserDebugParameter(self.target_velocity_direction_id) * 2 * 3.14159
            magnitude = p.readUserDebugParameter(self.target_velocity_magnitude_id)
            target_orientation_angle = p.readUserDebugParameter(self.target_orientation_id) * 2 * 3.14159
            
            # Set target velocity from sliders
            self.target_velocity = [magnitude * self.target_speed * np.cos(direction), 
                                   magnitude * self.target_speed * np.sin(direction), 0]
            # Set target orientation from slider (as quaternion)
            self.target_orientation = [0, 0, np.sin(target_orientation_angle / 2), np.cos(target_orientation_angle / 2)]
        except Exception as e:
            print(f"Warning: Could not read debug sliders in reset: {e}")
            # Keep the random values from parent reset
        
        return obs, info

    def step(self, action):
        # Read the debug parameters and set the target velocity and orientation accordingly
        try:
            direction = p.readUserDebugParameter(self.target_velocity_direction_id) * 2 * 3.14159  # 0 to 2pi
        except Exception as e:
            print(f"Error reading user debug parameter: {e}")
            direction = 0  # Default to 0 if there's an error
        try:
            magnitude = p.readUserDebugParameter(self.target_velocity_magnitude_id)  # 0 to 1
        except Exception as e:
            print(f"Error reading user debug parameter: {e}")
            magnitude = 0  # Default to 0 if there's an error
        try:
            target_orientation = p.readUserDebugParameter(self.target_orientation_id) * 2 * 3.14159  # 0 to 2pi
        except Exception as e:
            print(f"Error reading user debug parameter: {e}")
            target_orientation = 0  # Default to 0 if there's an error

        # Convert target velocity into 3d vector
        self.target_velocity = [magnitude * self.target_speed * np.cos(direction), magnitude * self.target_speed * np.sin(direction), 0]
        # Convert target orientation to quaternion [x, y, z, w]
        self.target_orientation = [0, 0, np.sin(target_orientation / 2), np.cos(target_orientation / 2)]
        
        # Debug: Print occasionally to verify values (every 100 steps)
        if self.steps_taken % 100 == 0:
            # Get current velocity and orientation for comparison
            current_vel, _ = p.getBaseVelocity(self.robot_id)
            current_pos, current_quat = p.getBasePositionAndOrientation(self.robot_id)
            current_yaw = p.getEulerFromQuaternion(current_quat)[2]
            target_yaw = target_orientation
            
            # Calculate velocity magnitude error
            vel_error = np.linalg.norm(np.array(current_vel[:2]) - np.array(self.target_velocity[:2]))
            
            # Calculate orientation (which way robot is facing) error
            yaw_error = abs(current_yaw - target_yaw)
            if yaw_error > np.pi:
                yaw_error = 2*np.pi - yaw_error
            
            # Calculate velocity DIRECTION error (which way robot is moving)
            target_vel_angle = np.arctan2(self.target_velocity[1], self.target_velocity[0])
            current_vel_angle = np.arctan2(current_vel[1], current_vel[0])
            vel_direction_error = abs(current_vel_angle - target_vel_angle)
            if vel_direction_error > np.pi:
                vel_direction_error = 2*np.pi - vel_direction_error
            
            print(f"Step {self.steps_taken}:")
            print(f"  Target: vel=[{self.target_velocity[0]:.3f}, {self.target_velocity[1]:.3f}] ({target_vel_angle:.2f} rad), face={target_orientation:.2f} rad")
            print(f"  Current: vel=[{current_vel[0]:.3f}, {current_vel[1]:.3f}] ({current_vel_angle:.2f} rad), face={current_yaw:.2f} rad")
            print(f"  Error: vel_mag={vel_error:.3f} m/s, vel_dir={vel_direction_error:.3f} rad, face_dir={yaw_error:.3f} rad")

        # Clear previous debug lines
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
        
        # Get current robot base position from PyBullet
        start_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Draw a line indicating the target velocity direction and magnitude
        # Length scales with velocity magnitude (multiply by a factor for visibility)
        velocity_scale = 2.0  # Make the line 2x longer than actual velocity for better visibility
        end_pos = [start_pos[0] + self.target_velocity[0] * velocity_scale, 
                   start_pos[1] + self.target_velocity[1] * velocity_scale, 
                   start_pos[2]]
        line_id = p.addUserDebugLine(start_pos, end_pos, [1, 0, 0], 3)  # Red line (thicker)
        self.debug_lines.append(line_id)
        
        # Draw a line indicating the target orientation direction (use the angle from the slider directly)
        # Fixed length for orientation (not scaled by velocity)
        orient_length = 0.5  # Fixed length in meters
        orient_end_pos = [start_pos[0] + orient_length * np.cos(target_orientation), 
                         start_pos[1] + orient_length * np.sin(target_orientation), 
                         start_pos[2]]
        orient_line_id = p.addUserDebugLine(start_pos, orient_end_pos, [0, 1, 0], 3)  # Green line (thicker)
        self.debug_lines.append(orient_line_id)

        # Get current velocity and orientation of the bot
        current_linear_vel, current_angular_vel = p.getBaseVelocity(self.robot_id)
        current_pos, current_orientation_quat = p.getBasePositionAndOrientation(self.robot_id)
        
        # Draw current velocity vector (dark red: [0.5, 0, 0])
        current_vel_end_pos = [current_pos[0] + current_linear_vel[0] * velocity_scale,
                               current_pos[1] + current_linear_vel[1] * velocity_scale,
                               current_pos[2]]
        current_vel_line_id = p.addUserDebugLine(current_pos, current_vel_end_pos, [0.5, 0, 0], 3)  # Dark red line
        self.debug_lines.append(current_vel_line_id)
        
        # Draw current orientation vector (dark green: [0, 0.5, 0])
        # Extract yaw angle from quaternion
        current_euler = p.getEulerFromQuaternion(current_orientation_quat)
        current_yaw = current_euler[2]  # z-axis rotation (yaw)
        current_orient_end_pos = [current_pos[0] + orient_length * np.cos(current_yaw),
                                  current_pos[1] + orient_length * np.sin(current_yaw),
                                  current_pos[2]]
        current_orient_line_id = p.addUserDebugLine(current_pos, current_orient_end_pos, [0, 0.5, 0], 3)  # Dark green line
        self.debug_lines.append(current_orient_line_id)

        return super().step(action)


if __name__ == "__main__":
    # To use a different robot, change the filename here

    urdf_file, save_path, save_prefix, model_path = utils.select_robot(load_model=True)

    min_z = get_min_z(urdf_file)
    # Create the environment. Stable-baselines will automatically call reset.
    env = VisualizationEnv(urdf_filename=urdf_file, start_position=[0, 0, -min_z])
    
    # Optionally wrap with video recording (comment out if you don't want videos)
    # env = wrappers.RecordVideo(env, video_folder='./videos/', name_prefix='servobot_demo', episode_trigger=lambda x: True)

    # check to make sure we didnt forget to import a model lmao
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train your model first using train.py.")
        exit(1)
    
    # Try to load the trained model
    try:
        model = PPO.load(model_path, env=env, device='cpu')
    except ValueError as e:
        if "Unexpected observation shape" in str(e):
            print(f"\n⚠️  ERROR: Model observation space mismatch!")
            print(f"The trained model expects a different observation space than the current environment.")
            print(f"This happens when you modify the environment after training.\n")
            print(f"SOLUTION: You need to retrain the model with the updated environment.")
            print(f"Run: python train.py\n")
            exit(1)
        else:
            raise


    print("Running trained model - adjust sliders to change target velocity/orientation")
    print("=" * 80)
    print("Controls:")
    print("  - Target Velocity Direction: 0=Right, 0.25=Up, 0.5=Left, 0.75=Down")
    print("  - Target Velocity Magnitude: 0=Stopped, 1=Full speed")
    print("  - Target Orientation: 0=Right, 0.25=Up, 0.5=Left, 0.75=Down")
    print()
    print("Visualization:")
    print("  - Light RED line: Target velocity")
    print("  - Light GREEN line: Target orientation")
    print("  - Dark RED line: Current velocity")
    print("  - Dark GREEN line: Current orientation")
    print("=" * 80)
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    episode_count = 0
    
    # Run indefinitely (press Ctrl+C to stop)
    try:
        while True:
            # Update targets from sliders BEFORE getting action
            # This ensures the policy sees the current slider values
            try:
                direction = p.readUserDebugParameter(env.target_velocity_direction_id) * 2 * 3.14159
                magnitude = p.readUserDebugParameter(env.target_velocity_magnitude_id)
                target_orientation_angle = p.readUserDebugParameter(env.target_orientation_id) * 2 * 3.14159
                
                env.target_velocity = [magnitude * env.target_speed * np.cos(direction), 
                                      magnitude * env.target_speed * np.sin(direction), 0]
                env.target_orientation = [np.cos(target_orientation_angle), np.sin(target_orientation_angle), 0]
            except:
                pass  # If reading fails, keep current targets
            
            # Get fresh observation with updated targets
            obs = env._get_obs()
            
            # Use the trained model to predict the next action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if done:
                episode_count += 1
                print(f"Episode {episode_count} finished with total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0
                done = False
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    
    env.close()
    print("Evaluation finished.")
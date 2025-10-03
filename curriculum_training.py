import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback 
import pybullet as p
import os 

class CurriculumTrainer:
    def __init__(self, env, model, config, save_path='./models/curriculum/'):
        self.env = env
        self.model = model
        self.config = config
        self.total_steps = 0
        self.savedir = save_path
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def run(self, total_steps):
        self.total_steps = total_steps
        for step_interval in range(total_steps//1000):
            
            phase_name, phase_config = self.get_curriculum_info(step_interval * 1000)
            
            if step_interval % 10 == 0: 
                print(f"Training Phase: {phase_name} | Steps: {step_interval * 1000} to {(step_interval + 1) * 1000}")
            self.env.update_config(phase_config)
            
            # Put each phase's models in a separate folder
            if phase_name == "Stand Still":
                save_path = self.savedir + 'stand_still/'
            elif phase_name == "Turn to Face Orientation":
                save_path = self.savedir + 'turn_orientation/'
            elif phase_name == "Movement":
                save_path = self.savedir + 'movement/'
            else:
                save_path = self.savedir + 'other/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(f"Saving models to: {save_path}")

            callback = CheckpointCallback(save_freq=5000, save_path=save_path, name_prefix='curriculum_model')
            self.model = self.model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=callback)

           
    def get_curriculum_info(self, step):
        """
        Get curriculum info for a specified step out of our total number of steps.
        Train the model in three distinct phases:
        1. Learn to stand still. Starts off with a ramping initial momentum it must counteract (increases in strength over time).
        2. Learn to turn and face a randomly generated orientation command. Still has random initial momentum. 
        3. Learn to walk in a given direction with a target speed. Still has random initial momentum.
        Each phase lasts for steps/3 timesteps.
        returns:
        Phase Name: string
        config: dict
        """
        config = self.config
        phase_duration = self.total_steps // 3
        phase_name = ""
        if step < phase_duration:
            phase_name = "Stand Still"
            # Phase 1: Learn to stand still against initial momentum
            config['TARGET_SPEED'] = 0.0
            config['FORWARD_VEL_WEIGHT'] = 0.0
            config['ORIENTATION_REWARD_WEIGHT'] = 0.0  # No orientation weight in this phase
            config['INITIAL_MOMENTUM'] = min(1.0, step / phase_duration)  # Ramp up initial momentum from 0 to 1
        elif step < 2 * phase_duration:
            phase_name = "Turn to Face Orientation"
            # Phase 2: Learn to turn to face a random orientation command
            config['TARGET_SPEED'] = 0.0
            config['FORWARD_VEL_WEIGHT'] = 0.0
            config['ORIENTATION_REWARD_WEIGHT'] = 1.0  # Full orientation weight
            config['INITIAL_MOMENTUM'] = 1.0  # Full initial momentum
        else:
            phase_name = "Movement"
            # Phase 3: Learn to walk in a given direction with target speed
            if step % 1000 == 0:  # increase target speed every 1000 steps
                config['TARGET_SPEED'] = min(1.0, config.get('TARGET_SPEED', 0.0) + 0.1)  # Increase target speed up to 1.0
            config['FORWARD_VEL_WEIGHT'] = 100.0  # Full forward velocity weight
            config['ORIENTATION_REWARD_WEIGHT'] = 1.0  # Full orientation weight
            config['INITIAL_MOMENTUM'] = 1.0  # Full initial momentum

        return phase_name, config

if __name__ == "__main__":
    from src.envs import env
    from src.utils import utils

    urdf_file, save_path, save_prefix = utils.select_robot(load_model = False)
    min_z = env.get_min_z(urdf_file)
    base_env = env.BaseEnv(render_mode='headless', urdf_filename=urdf_file, start_position=[0, 0, -min_z])

    model = PPO("MlpPolicy", base_env, verbose=1, n_steps=2048)

    # Initial configuration for curriculum training
    initial_config = {
        # Add other config parameters as needed
    }

    trainer = CurriculumTrainer(base_env, model, initial_config, save_path=save_path)
    trainer.run(total_steps=300000)  # Total of 300k steps for curriculum training
    
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback 
import pybullet as p
import os 

class CurriculumTrainer:
    def __init__(self, env, model, config):
        self.env = env
        self.model = model
        self.config = config
        self.total_steps = 0
    

    def run(self, total_steps):
        self.total_steps = total_steps
        for step_interval in range(total_steps//10000):
            
            phase_name, phase_config = self.get_curriculum_info(step_interval * 10000)
            print(f"Training Phase: {phase_name} | Steps: {step_interval * 10000} to {(step_interval + 1) * 10000}")
            self.env.update_config(phase_config)
            
            # Put each phase's models in a separate folder
            if phase_name == "Stand Still":
                save_path = './models/stand_still/'
            elif phase_name == "Turn to Face Orientation":
                save_path = './models/turn_orientation/'
            elif phase_name == "Movement":
                save_path = './models/movement/'
            else:
                save_path = './models/other/'
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
            config['target_speed'] = 0.0
            config['orientation_command'] = None
            config['initial_momentum'] = min(1.0, step / phase_duration)  # Ramp up initial momentum from 0 to 1
        elif step < 2 * phase_duration:
            phase_name = "Turn to Face Orientation"
            # Phase 2: Learn to turn to face a random orientation command
            config['target_speed'] = 0.0
            if step % 1000 == 0:  # Change orientation command every 1000 steps
                config['orientation_command'] = np.random.uniform(-np.pi, np.pi)
            config['initial_momentum'] = 1.0  # Full initial momentum
        else:
            phase_name = "Movement"
            # Phase 3: Learn to walk in a given direction with target speed
            if step % 1000 == 0:  # Change target speed every 1000 steps
                config['target_speed'] = np.random.uniform(0.2, 1.0)  # Random target speed between 0.2 and 1.0 m/s
            config['orientation_command'] = None
            config['initial_momentum'] = 1.0  # Full initial momentum

        return phase_name, config

if __name__ == "__main__":
    from src.envs import env
    from src.utils import utils

    urdf_file, save_path, save_prefix, model_path = utils.select_robot()
    min_z = env.get_min_z(urdf_file)
    base_env = env.BaseEnv(render_mode='human', urdf_filename=urdf_file, start_position=[0, 0, -min_z])

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train your model first using train.py.")
        exit(1)
    else:
        # Load the trained model
        model = PPO.load(model_path, env=base_env, device='cpu')

    # Initial configuration for curriculum training
    initial_config = {
        'target_speed': 0.0,
        'orientation_command': None,
        'initial_momentum': 0.0,
        # Add other config parameters as needed
    }

    trainer = CurriculumTrainer(base_env, model, initial_config)
    trainer.run(total_steps=300000)  # Total of 300k steps for curriculum training
    
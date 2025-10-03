import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback 
import pybullet as p

class CurriculumTrainer:
    def __init__(self, env, model, config):
        self.env = env
        self.model = model
        self.config = config
        self.total_steps = 0
    

    def run(self, total_steps):
        self.total_steps = total_steps
        callback = CheckpointCallback(save_freq=10000, 
                                      save_path=self.config['save_path'], 
                                      name_prefix=self.config['save_prefix'])
        self.model.learn(total_timesteps=10000, reset_num_timesteps=False, callback=callback)

    def visualize_model(self, episodes=3):
        print("Running trained model for 3 episodes...")
        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Use the trained model to predict the next action
                action, _states = model.predict(obs, deterministic=True)
                
                # Take the action in the environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                
            print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
    def get_curriculum_weights(self, steps):
        """
        Run curriculum training for a specified number of steps.
        Train the model in three distinct phases:
        1. Learn to stand still. Starts off with a ramping initial momentum it must counteract (increases in strength over time).
        2. Learn to turn and face a randomly generated orientation command. Still has random initial momentum. 
        3. Learn to walk in a given direction with a target speed. Still has random initial momentum.
        Each phase lasts for steps/3 timesteps.
        """
        phase_duration = steps // 3
        try:
            for step in range(steps):
                if step < phase_duration:
                    # Phase 1: Learn to stand still against initial momentum
                    config['target_speed'] = 0.0
                    config['orientation_command'] = None
                    config['initial_momentum'] = min(1.0, step / phase_duration)  # Ramp up initial momentum from 0 to 1
                elif step < 2 * phase_duration:
                    # Phase 2: Learn to turn to face a random orientation command
                    config['target_speed'] = 0.0
                    if step % 1000 == 0:  # Change orientation command every 1000 steps
                        config['orientation_command'] = np.random.uniform(-np.pi, np.pi)
                    config['initial_momentum'] = 1.0  # Full initial momentum
                else:
                    # Phase 3: Learn to walk in a given direction with target speed
                    if step % 1000 == 0:  # Change target speed every 1000 steps
                        config['target_speed'] = np.random.uniform(0.2, 1.0)  # Random target speed between 0.2 and 1.0 m/s
                    config['orientation_command'] = None
                    config['initial_momentum'] = 1.0  # Full initial momentum
                
                # Train the model for one step with the current configuration
                model.learn(total_timesteps=1, reset_num_timesteps=False)
                if step % 10000 == 0:
                    print(f"Completed {step} / {steps} steps of curriculum training.")
            
            print("Curriculum training completed.")
            model.save(config['save_path'] + config['save_prefix'] + "_curriculum_trained")
        except KeyboardInterrupt:
            print("Training interrupted by user. Saving model...")
            model.save(config['save_path'] + config['save_prefix'] + "_interrupted")
            print("Model saved. Exiting.")



    
"""
100% vibe coded callback thing but lowkey works really well so not touching it 
Custom callback for real-time training visualization
Shows plots of key metrics that update during training
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque


class LivePlottingCallback(BaseCallback):
    """
    Callback that creates live plots of training metrics
    Updates plots periodically during training
    """
    
    def __init__(self, plot_freq=2048, max_points=500, verbose=0):
        """
        Args:
            plot_freq: Update plot every N steps
            max_points: Maximum number of points to display (for performance)
            verbose: Verbosity level
        """
        super(LivePlottingCallback, self).__init__(verbose)
        self.plot_freq = plot_freq
        self.max_points = max_points
        
        # Storage for metrics
        self.timesteps = deque(maxlen=max_points)
        self.ep_len_mean = deque(maxlen=max_points)
        self.ep_rew_mean = deque(maxlen=max_points)
        self.explained_var = deque(maxlen=max_points)
        self.value_loss = deque(maxlen=max_points)
        self.policy_loss = deque(maxlen=max_points)
        self.approx_kl = deque(maxlen=max_points)
        self.fps_data = deque(maxlen=max_points)
        
        # Plot setup
        self.fig = None
        self.axes = None
        self.lines = {}
        self.plot_initialized = False
        
    def _init_plots(self):
        """Initialize the plot window"""
        plt.ion()  # Enable interactive mode
        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.fig.suptitle('Real-Time Training Metrics', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        self.axes = self.axes.flatten()
        
        # Configure each subplot
        plot_configs = [
            ('Episode Length', 'Steps', 'Episode Length', 0),
            ('Episode Reward', 'Reward', 'Episode Reward', 1),
            ('Explained Variance', 'Variance', 'Explained Variance', 2),
            ('Value Loss', 'Loss', 'Value Loss', 3),
            ('Policy Loss', 'Loss', 'Policy Gradient Loss', 4),
            ('Approx KL', 'KL Divergence', 'KL Divergence', 5),
            ('Training Speed', 'FPS', 'Frames per Second', 6),
            ('Timesteps', 'Timesteps', 'Total Timesteps', 7),
        ]
        
        for title, ylabel, label, idx in plot_configs:
            ax = self.axes[idx]
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            line, = ax.plot([], [], 'b-', linewidth=2, label=label)
            self.lines[label] = line
            ax.legend(loc='upper left', fontsize=8)
        
        plt.tight_layout()
        self.plot_initialized = True
        
    def _update_plots(self):
        """Update all plots with current data"""
        if not self.plot_initialized:
            return
        
        # Convert deques to lists for plotting
        x = list(self.timesteps)
        
        # Update each plot
        if len(x) > 0:
            # Episode Length
            self.lines['Episode Length'].set_data(x, list(self.ep_len_mean))
            self.axes[0].relim()
            self.axes[0].autoscale_view()
            
            # Episode Reward
            self.lines['Episode Reward'].set_data(x, list(self.ep_rew_mean))
            self.axes[1].relim()
            self.axes[1].autoscale_view()
            
            # Explained Variance
            self.lines['Explained Variance'].set_data(x, list(self.explained_var))
            self.axes[2].relim()
            self.axes[2].autoscale_view()
            self.axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            self.axes[2].axhline(y=1, color='g', linestyle='--', alpha=0.5, linewidth=1)
            
            # Value Loss
            self.lines['Value Loss'].set_data(x, list(self.value_loss))
            self.axes[3].relim()
            self.axes[3].autoscale_view()
            
            # Policy Loss
            self.lines['Policy Gradient Loss'].set_data(x, list(self.policy_loss))
            self.axes[4].relim()
            self.axes[4].autoscale_view()
            
            # Approx KL
            self.lines['KL Divergence'].set_data(x, list(self.approx_kl))
            self.axes[5].relim()
            self.axes[5].autoscale_view()
            self.axes[5].axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Warning')
            self.axes[5].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Danger')
            
            # FPS
            self.lines['Frames per Second'].set_data(x, list(self.fps_data))
            self.axes[6].relim()
            self.axes[6].autoscale_view()
            
            # Progress (total timesteps)
            self.lines['Total Timesteps'].set_data(x, x)
            self.axes[7].relim()
            self.axes[7].autoscale_view()
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _on_step(self) -> bool:
        """
        Called after each step in the environment
        Returns True to continue training, False to stop
        """
        # Only update plots periodically
        if self.n_calls % self.plot_freq == 0:
            # Initialize plots on first call
            if not self.plot_initialized:
                self._init_plots()
            
            # Get metrics from logger
            if len(self.model.ep_info_buffer) > 0:
                # Episode metrics (from rollout)
                ep_len = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                ep_rew = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                
                self.timesteps.append(self.num_timesteps)
                self.ep_len_mean.append(ep_len)
                self.ep_rew_mean.append(ep_rew)
                
                # Training metrics (from logger)
                if hasattr(self.logger, 'name_to_value'):
                    # Explained variance
                    if 'train/explained_variance' in self.logger.name_to_value:
                        self.explained_var.append(self.logger.name_to_value['train/explained_variance'])
                    elif len(self.explained_var) > 0:
                        self.explained_var.append(self.explained_var[-1])  # Repeat last value
                    else:
                        self.explained_var.append(0.0)
                    
                    # Value loss
                    if 'train/value_loss' in self.logger.name_to_value:
                        self.value_loss.append(self.logger.name_to_value['train/value_loss'])
                    elif len(self.value_loss) > 0:
                        self.value_loss.append(self.value_loss[-1])
                    else:
                        self.value_loss.append(0.0)
                    
                    # Policy loss
                    if 'train/policy_gradient_loss' in self.logger.name_to_value:
                        self.policy_loss.append(self.logger.name_to_value['train/policy_gradient_loss'])
                    elif len(self.policy_loss) > 0:
                        self.policy_loss.append(self.policy_loss[-1])
                    else:
                        self.policy_loss.append(0.0)
                    
                    # Approx KL
                    if 'train/approx_kl' in self.logger.name_to_value:
                        self.approx_kl.append(self.logger.name_to_value['train/approx_kl'])
                    elif len(self.approx_kl) > 0:
                        self.approx_kl.append(self.approx_kl[-1])
                    else:
                        self.approx_kl.append(0.0)
                    
                    # FPS
                    if 'time/fps' in self.logger.name_to_value:
                        self.fps_data.append(self.logger.name_to_value['time/fps'])
                    elif len(self.fps_data) > 0:
                        self.fps_data.append(self.fps_data[-1])
                    else:
                        self.fps_data.append(0.0)
                
                # Update the plots
                self._update_plots()
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.plot_initialized:
            print("\nTraining completed! Close the plot window to continue...")
            plt.ioff()  # Disable interactive mode
            plt.show()  # Keep plot window open


class LivePlottingCallbackNoGUI(BaseCallback):
    """
    Callback for headless training - saves plots periodically instead of showing them
    """
    
    def __init__(self, plot_freq=10000, save_freq=50000, save_path='./training_plots/', verbose=0):
        super(LivePlottingCallbackNoGUI, self).__init__(verbose)
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        self.save_path = save_path
        
        # Storage for metrics
        self.timesteps = []
        self.ep_len_mean = []
        self.ep_rew_mean = []
        self.explained_var = []
        self.value_loss = []
        self.policy_loss = []
        self.approx_kl = []
        self.fps_data = []
        
        # Create save directory
        import os
        os.makedirs(save_path, exist_ok=True)
        
    def _save_plots(self):
        """Save current plots to file"""
        if len(self.timesteps) == 0:
            return
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Training Metrics at {self.num_timesteps} Steps', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Plot each metric
        axes[0].plot(self.timesteps, self.ep_len_mean, 'b-', linewidth=2)
        axes[0].set_title('Episode Length')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Steps')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.timesteps, self.ep_rew_mean, 'b-', linewidth=2)
        axes[1].set_title('Episode Reward')
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Reward')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(self.timesteps, self.explained_var, 'b-', linewidth=2)
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].axhline(y=1, color='g', linestyle='--', alpha=0.5)
        axes[2].set_title('Explained Variance')
        axes[2].set_xlabel('Timesteps')
        axes[2].set_ylabel('Variance')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(self.timesteps, self.value_loss, 'b-', linewidth=2)
        axes[3].set_title('Value Loss')
        axes[3].set_xlabel('Timesteps')
        axes[3].set_ylabel('Loss')
        axes[3].grid(True, alpha=0.3)
        
        axes[4].plot(self.timesteps, self.policy_loss, 'b-', linewidth=2)
        axes[4].set_title('Policy Loss')
        axes[4].set_xlabel('Timesteps')
        axes[4].set_ylabel('Loss')
        axes[4].grid(True, alpha=0.3)
        
        axes[5].plot(self.timesteps, self.approx_kl, 'b-', linewidth=2)
        axes[5].axhline(y=0.02, color='orange', linestyle='--', alpha=0.5)
        axes[5].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
        axes[5].set_title('Approx KL')
        axes[5].set_xlabel('Timesteps')
        axes[5].set_ylabel('KL Divergence')
        axes[5].grid(True, alpha=0.3)
        
        axes[6].plot(self.timesteps, self.fps_data, 'b-', linewidth=2)
        axes[6].set_title('Training Speed')
        axes[6].set_xlabel('Timesteps')
        axes[6].set_ylabel('FPS')
        axes[6].grid(True, alpha=0.3)
        
        axes[7].plot(self.timesteps, self.timesteps, 'b-', linewidth=2)
        axes[7].set_title('Progress (Very Important Metric)')
        axes[7].set_xlabel('Timesteps')
        axes[7].set_ylabel('Total Steps')
        axes[7].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_file = f'{self.save_path}/training_metrics_{self.num_timesteps}.png'
        plt.savefig(save_file, dpi=100)
        plt.close(fig)
        
        if self.verbose > 0:
            print(f"Saved training plot to: {save_file}")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_len = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                ep_rew = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                
                self.timesteps.append(self.num_timesteps)
                self.ep_len_mean.append(ep_len)
                self.ep_rew_mean.append(ep_rew)
                
                if hasattr(self.logger, 'name_to_value'):
                    self.explained_var.append(self.logger.name_to_value.get('train/explained_variance', 0.0))
                    self.value_loss.append(self.logger.name_to_value.get('train/value_loss', 0.0))
                    self.policy_loss.append(self.logger.name_to_value.get('train/policy_gradient_loss', 0.0))
                    self.approx_kl.append(self.logger.name_to_value.get('train/approx_kl', 0.0))
                    self.fps_data.append(self.logger.name_to_value.get('time/fps', 0.0))
        
        # Save plots periodically
        if self.n_calls % self.save_freq == 0:
            self._save_plots()
        
        return True
    
    def _on_training_end(self) -> None:
        """Save final plot"""
        self._save_plots()
        print(f"Final training plots saved to: {self.save_path}")

from .env import BaseEnv

class QuadrupedEnv(BaseEnv):
    def __init__(self, render_mode=None, urdf_filename="simple_quadruped.urdf", start_position=[0, 0, 1],
                    target_box_center=[10.0, 0.0], target_box_size=[1.0, 1.0, 1.0]):
            super(QuadrupedEnv, self).__init__(render_mode, urdf_filename, start_position, target_box_center, target_box_size)

    def step(self, action):
          return super().step(action)
    
    def reset(self, seed=None, options=None):
        return super().reset(seed, options)
    
    def _get_obs(self):
        return super()._get_obs()
    
    def render(self):
        return super().render()
    
    def close(self):
        return super().close()
    
    def train(self):
        return super().train()

    def evaluate(self):
        return super().evaluate()
    
    def compute_reward(self):
        return super().compute_reward()
    
    def _is_success(self):
        return super()._is_success()
    
    def _get_info(self):
        return super()._get_info()
    

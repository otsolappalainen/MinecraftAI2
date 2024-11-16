import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import pickle

class LoggingCallback(BaseCallback):
    """
    Custom callback for logging observations, actions, and rewards during training.
    """
    def __init__(self, log_dir, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.observations = []
        self.actions = []
        self.rewards = []
        self.episode_starts = []

    def _on_step(self) -> bool:
        # Get the local variables from the training loop
        obs = self.locals.get('new_obs')
        action = self.locals.get('action')
        reward = self.locals.get('rewards')
        done = self.locals.get('dones')

        # Append data to lists
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.episode_starts.append(done)

        return True  # Continue training

    def _on_training_end(self):
        # Save the logged data to files
        with open(os.path.join(self.log_dir, 'observations.pkl'), 'wb') as f:
            pickle.dump(self.observations, f)
        np.save(os.path.join(self.log_dir, 'actions.npy'), self.actions)
        np.save(os.path.join(self.log_dir, 'rewards.npy'), self.rewards)
        np.save(os.path.join(self.log_dir, 'episode_starts.npy'), self.episode_starts)
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import pickle

class LoggingCallback(BaseCallback):
    """
    Custom callback for logging observations, actions, and rewards during training.
    """
    def __init__(self, log_dir, reward_clamp=(-10, 10), verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.reward_clamp = reward_clamp  # Define the reward clamping range
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize logs
        self.observations = []
        self.actions = []
        self.raw_rewards = []
        self.clamped_rewards = []
        self.cumulative_rewards = []
        self.episode_lengths = []

        # Episode tracking
        self.current_cumulative_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # Get the local variables from the training loop
        obs = self.locals.get('new_obs', None)
        action = self.locals.get('action', None)
        reward = self.locals.get('rewards', None)
        done = self.locals.get('dones', None)

        # Logging observations
        if obs is not None:
            self.observations.append(obs)

        # Logging actions
        if action is not None:
            self.actions.append(action)

        # Logging rewards
        if reward is not None:
            self.raw_rewards.append(reward)
            clamped_reward = np.clip(reward, *self.reward_clamp)  # Clamp the reward
            self.clamped_rewards.append(clamped_reward)
            self.current_cumulative_reward += clamped_reward

        # End of episode
        if done is not None and done[0]:  # Assuming done[0] indicates episode termination
            self.cumulative_rewards.append(self.current_cumulative_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_cumulative_reward = 0
            self.current_episode_length = 0
        else:
            self.current_episode_length += 1

        return True  # Continue training

    def _on_training_end(self):
        # Save the logged data to files
        with open(os.path.join(self.log_dir, 'observations.pkl'), 'wb') as f:
            pickle.dump(self.observations, f)

        np.save(os.path.join(self.log_dir, 'actions.npy'), self.actions)
        np.save(os.path.join(self.log_dir, 'raw_rewards.npy'), self.raw_rewards)
        np.save(os.path.join(self.log_dir, 'clamped_rewards.npy'), self.clamped_rewards)
        np.save(os.path.join(self.log_dir, 'cumulative_rewards.npy'), self.cumulative_rewards)
        np.save(os.path.join(self.log_dir, 'episode_lengths.npy'), self.episode_lengths)
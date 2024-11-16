# analyze_training_data.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def analyze_training_data(log_dir=r'E:\training data\fullAIlogs'):
    # Load the logged data
    with open(os.path.join(log_dir, 'observations.pkl'), 'rb') as f:
        observations = pickle.load(f)
    actions = np.load(os.path.join(log_dir, 'actions.npy'), allow_pickle=True)
    rewards = np.load(os.path.join(log_dir, 'rewards.npy'), allow_pickle=True)
    episode_starts = np.load(os.path.join(log_dir, 'episode_starts.npy'), allow_pickle=True)

    # Convert actions and rewards to arrays
    actions = np.array(actions).flatten()
    rewards = np.array(rewards).flatten()
    episode_starts = np.array(episode_starts).flatten()

    # 1. Plot cumulative reward over time
    cumulative_rewards = np.cumsum(rewards)
    plt.figure()
    plt.plot(cumulative_rewards)
    plt.title('Cumulative Reward over Time')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.show()

    # 2. Plot action distribution
    actions = np.array([action for action in actions if action is not None]).flatten()
    if actions.size > 0:
        unique_actions, counts = np.unique(actions, return_counts=True)
        plt.figure()
        plt.bar(unique_actions, counts)
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.show()
    else:
        print("No valid actions to analyze.")

    # 3. Analyze positions over time
    # Analyze positions over time
    positions = [obs['position'] for obs in observations if 'position' in obs and isinstance(obs['position'], (list, np.ndarray))]
    positions = np.array(positions)

    # Check if positions need reshaping
    if positions.ndim == 3 and positions.shape[1] == 1 and positions.shape[2] == 5:
        # Flatten to shape (n_samples, 5)
        positions = positions[:, 0, :]

    if positions.ndim == 2 and positions.shape[1] >= 3:
        # Plot if positions have at least three dimensions (X, Y, Z)
        plt.figure()
        plt.plot(positions[:, 0], label='X')
        plt.plot(positions[:, 1], label='Y')
        plt.plot(positions[:, 2], label='Z')
        plt.title('Position over Time')
        plt.xlabel('Step')
        plt.ylabel('Position')
        plt.legend()
        plt.show()
    else:
        print(f"Positions array has unexpected shape: {positions.shape}. Skipping position analysis.")

    # 4. Analyze rewards per episode
    episode_rewards = []
    current_reward = 0
    for reward, done in zip(rewards, episode_starts):
        current_reward += reward
        if done:
            episode_rewards.append(current_reward)
            current_reward = 0

    plt.figure()
    plt.plot(episode_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    # Add more analysis as needed

if __name__ == '__main__':
    analyze_training_data()

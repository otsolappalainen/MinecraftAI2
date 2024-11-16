import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

# Function to list and select a file
def select_file(directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface"):
    # List all CSV files in the specified directory
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified directory.")
        return None

    # Display the list of files
    print("Available CSV files:")
    for i, file_name in enumerate(csv_files, 1):
        print(f"{i}: {file_name}")

    # Ask the user to select a file
    file_index = int(input("Select a file by entering its number: ")) - 1

    # Return the full path of the selected file
    return os.path.join(directory, csv_files[file_index]) if 0 <= file_index < len(csv_files) else None

# Function to load episode rewards from a CSV file
def load_episode_rewards(file_path):
    # Use a dictionary to store rewards by trial number
    episode_data = defaultdict(list)

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Ensure the row has at least three columns (trial, episode, reward)
            if len(row) >= 3:
                try:
                    trial = int(row[0])    # First column is trial number
                    episode = int(row[1])  # Second column is episode number (x-axis)
                    reward = float(row[2]) # Third column is reward (y-axis)
                    episode_data[trial].append((episode, reward))
                except ValueError:
                    print(f"Skipping row with unexpected format: {row}")

    return episode_data

# Function to calculate rolling average
def calculate_rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to plot rewards over episodes for each trial
def plot_reward_progress_for_trials(episode_data, window_sizes=[50]):
    for trial_index, rewards in episode_data.items():
        rewards = sorted(rewards)  # Sort by episode number to ensure correct x-axis ordering
        episodes, reward_values = zip(*rewards)  # Separate episodes and rewards

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, reward_values, label='Reward')

        # Plot rolling averages for each specified window size
        for window_size in window_sizes:
            avg_rewards = calculate_rolling_average(reward_values, window_size)
            plt.plot(episodes[window_size - 1:], avg_rewards, label=f'Avg {window_size} Episodes')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward Progression for Trial {trial_index} (with Rolling Averages)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Ask the user to select a CSV file
    selected_file = select_file()
    
    if selected_file:
        print(f"Loading data from: {selected_file}")
        # Load episode rewards from the selected file
        episode_data = load_episode_rewards(selected_file)
        if episode_data:
            plot_reward_progress_for_trials(episode_data)
    else:
        print("No file selected.")
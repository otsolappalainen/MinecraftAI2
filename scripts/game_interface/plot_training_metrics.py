import matplotlib.pyplot as plt
import numpy as np
import re
import os

# List all log files with "training_log" in the name
log_files = [file for file in os.listdir() if "training_log" in file and file.endswith(".txt")]

# Check if any log files are found
if not log_files:
    print("No training log files found in the current directory.")
else:
    # Display the list of files for the user to select
    print("Available training log files:")
    for i, file in enumerate(log_files):
        print(f"{i + 1}. {file}")

    # Prompt the user to pick a file by entering a number
    selected_index = int(input("Enter the number of the file you want to load: ")) - 1
    log_file = log_files[selected_index]
    print(f"Selected file: {log_file}")

    # Load log data from the selected file
    with open(log_file, 'r') as file:
        log_data = file.readlines()

    # Parse cumulative rewards and distances from log
    cumulative_rewards = []
    distances = []

    for line in log_data:
        if "Cumulative Reward:" in line:
            reward = float(re.search(r"Cumulative Reward: ([\-\d\.]+)", line).group(1))
            cumulative_rewards.append(reward)
        elif "at distance:" in line:
            distance = float(re.search(r"at distance: ([\d\.]+)", line).group(1))
            distances.append(distance)

    # Convert to numpy arrays for easier calculations
    cumulative_rewards = np.array(cumulative_rewards)
    distances = np.array(distances)

    # Calculate rolling average for cumulative rewards (average over every 20 episodes)
    avg_cumulative_rewards = np.convolve(cumulative_rewards, np.ones(20) / 20, mode='valid')

    # Calculate agents that reached the target (distance <= 10 considered as reaching target)
    reached_target = distances <= 10
    agents_reaching_target = np.cumsum(reached_target)

    # Adjust distances to set 10 for agents that reached the target
    distances = np.where(reached_target, 10, distances)

    # Plot cumulative reward over time (averaged over every 20 episodes)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_cumulative_rewards, label='Avg Cumulative Reward (20-episode avg)')
    plt.xlabel("Episode (in twenties)")
    plt.ylabel("Average Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.legend()
    plt.show()

    # Plot agents reaching the target per episode
    plt.figure(figsize=(12, 6))
    plt.plot(agents_reaching_target, label='Agents Reaching Target')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Agents Reaching Target")
    plt.title("Agents Reaching Target Per Episode")
    plt.legend()
    plt.show()

    # Plot average distance from target per episode
    avg_distances = np.convolve(distances, np.ones(10) / 10, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(avg_distances, label='Average Distance from Target (10-episode avg)')
    plt.xlabel("Episode (in tens)")
    plt.ylabel("Average Distance from Target")
    plt.title("Average Distance from Target Over Time")
    plt.legend()
    plt.show()
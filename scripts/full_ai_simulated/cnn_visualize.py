import csv
import os
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datetime import datetime

# Path to the training logs
LOG_FILE = r"E:\CNN\training_data.csv"
OUTPUT_DIR = r"E:\CNN\visualizations"  # Directory to save rendered videos or gifs


def read_log_file(log_file):
    """
    Read the training data from the CSV file, grouping episodes by step counts.
    """
    episodes = defaultdict(list)
    current_episode = []
    current_target = None

    with open(log_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Check if any critical fields are missing or None
                required_fields = ["step", "x", "z", "yaw", "reward", "task_x", "task_z"]
                if any(row.get(field) in (None, "") for field in required_fields):
                    print(f"Skipping invalid row: {row}")
                    continue

                # Convert values to appropriate types
                step = int(float(row["step"]))
                task_x = float(row["task_x"])
                task_z = float(row["task_z"])
                task_key = (task_x, task_z)

                # Detect a new episode when step count resets or reaches a threshold
                if step == 0 and current_episode:
                    episodes[current_target].append(current_episode)
                    current_episode = []

                # Update the current target and add the row to the current episode
                current_target = task_key
                current_episode.append({
                    "step": step,
                    "x": float(row["x"]),
                    "z": float(row["z"]),
                    "yaw": float(row["yaw"]),
                    "reward": float(row["reward"]),
                    "task_x": task_x,
                    "task_z": task_z,
                })
            except ValueError as e:
                print(f"Error parsing row {row}: {e}")
            except TypeError as e:
                print(f"Type error with row {row}: {e}")

    # Add the last episode if not empty
    if current_episode:
        episodes[current_target].append(current_episode)

    return episodes


def visualize_group(target_key, episodes, animation_speed=50, save_as_video=False):
    """
    Visualize a group of episodes with the same target as moving dots and show the target direction.
    """
    print(f"Visualizing group with target: {target_key}")

    # Extract data for each episode
    episode_positions = []
    for episode in episodes:
        x_positions = [data["x"] for data in episode]
        z_positions = [data["z"] for data in episode]
        episode_positions.append((x_positions, z_positions))

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Target Direction: {target_key}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.grid(True)

    # Draw the target direction
    # Normalize the target vector for direction
    target_x, target_z = target_key
    magnitude = (target_x**2 + target_z**2)**0.5
    normalized_x, normalized_z = target_x / magnitude, target_z / magnitude

    # Extend the target direction across the plot
    arrow_length = max(10, magnitude * 10)  # Make it visually significant
    ax.arrow(0, 0, normalized_x * arrow_length, normalized_z * arrow_length,
             head_width=2, head_length=3, fc='red', ec='red', label="Target Direction")

    # Initialize agent positions as scatter plot
    agents = ax.scatter([], [], color="blue", label="Agent Position")

    # Set plot limits dynamically based on all episodes
    all_x = [x for positions in episode_positions for x in positions[0]]
    all_z = [z for positions in episode_positions for z in positions[1]]
    margin = 10
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    ax.legend()

    def update(frame):
        """
        Update the scatter plot for each frame.
        """
        x_coords = []
        z_coords = []
        for x_positions, z_positions in episode_positions:
            if frame < len(x_positions):
                x_coords.append(x_positions[frame])
                z_coords.append(z_positions[frame])
        agents.set_offsets(list(zip(x_coords, z_coords)))
        return agents,

    ani = FuncAnimation(
        fig, update, frames=max(len(positions[0]) for positions in episode_positions), interval=animation_speed, blit=True
    )

    if save_as_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        total_steps = sum(len(ep) for ep in episodes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"target_{target_key[0]}_{target_key[1]}_{total_steps}_{timestamp}.mp4")
        ani.save(output_path, writer="ffmpeg")
        print(f"Saved video to {output_path}")
    else:
        plt.show()


def main():
    """
    Main function to run the visualization tool.
    """
    # Ensure the log file exists
    if not os.path.exists(LOG_FILE):
        print(f"Log file '{LOG_FILE}' not found.")
        return

    # Read the log data
    grouped_sessions = read_log_file(LOG_FILE)

    # Display available target groups
    print("\nAvailable Target Groups:")
    target_keys = list(grouped_sessions.keys())
    for i, key in enumerate(target_keys):
        print(f"{i}: Target {key}")

    choice = int(input("Enter the number of the target group to visualize: "))
    if 0 <= choice < len(target_keys):
        selected_target = target_keys[choice]
        selected_episodes = grouped_sessions[selected_target]
    else:
        print("Invalid choice. Exiting.")
        return

    # Ask how many episodes to visualize
    num_episodes = int(input(f"Enter the number of episodes to visualize (1-{len(selected_episodes)}): "))
    if num_episodes > len(selected_episodes):
        print("Requested episodes exceed available. Showing all.")
        num_episodes = len(selected_episodes)

    # Determine generation size and prompt user
    num_generations = (len(selected_episodes) + num_episodes - 1) // num_episodes  # Calculate ceil(len/num)
    if num_generations > 1:
        print(f"Generations available: 1-{num_generations}")
        generation_choice = input("Random sample (r) or select generation (1-N)? ").lower()
    else:
        generation_choice = "1"  # Only one generation available

    if generation_choice == "r":
        # Select a random sample of episodes
        sampled_episodes = random.sample(selected_episodes, num_episodes)
    else:
        try:
            generation = int(generation_choice)
            if 1 <= generation <= num_generations:
                start_idx = (generation - 1) * num_episodes
                end_idx = start_idx + num_episodes
                sampled_episodes = selected_episodes[start_idx:end_idx]
            else:
                print("Invalid generation choice. Selecting random episodes.")
                sampled_episodes = random.sample(selected_episodes, num_episodes)
        except ValueError:
            print("Invalid input. Selecting random episodes.")
            sampled_episodes = random.sample(selected_episodes, num_episodes)

    # Visualize the selected group
    save_as_video = input("Save as video? (yes/no, default no): ").lower() in ["yes", "y"]
    animation_speed = int(input("Enter animation speed (default 50ms per frame): ") or "50")

    visualize_group(selected_target, sampled_episodes, animation_speed, save_as_video)


if __name__ == "__main__":
    main()

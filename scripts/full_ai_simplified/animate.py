import csv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datetime import datetime
import numpy as np

# Directory containing the environment log files
LOG_DIR = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai_simplified"
OUTPUT_DIR = r"visualizations"  # Directory to save rendered videos or gifs


def is_valid_row(row):
    """
    Validate a row based on the given criteria.
    """
    try:
        step = int(float(row[1]))
        if step % 5 != 0 or step < 0 or step > 500:
            return False
        float(row[2])  # x
        float(row[3])  # z
        float(row[4])  # yaw
        float(row[5])  # reward
        task_x = int(float(row[6]))
        task_z = int(float(row[7]))
        if task_x not in [-1, 0, 1] or task_z not in [-1, 0, 1]:
            return False
        return True
    except (ValueError, IndexError):
        return False


def read_all_log_files(log_dir):
    """
    Read and aggregate data from all matching log files in the directory.
    """
    episodes = defaultdict(list)

    for filename in os.listdir(log_dir):
        if not filename.startswith("training_data_env_") or not filename.endswith(".csv"):
            continue  # Skip files that don't match the pattern
        file_path = os.path.join(log_dir, filename)
        print(f"Processing file: {file_path}")
        episodes = read_single_log_file(file_path, episodes)
    return episodes


def read_single_log_file(log_file, episodes):
    """
    Read a single log file and update the episodes dictionary.
    """
    current_episode = []
    current_target = None
    last_step = -1

    with open(log_file, mode="r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not is_valid_row(row):
                continue
            try:
                step = int(float(row[1]))
                task_x = int(float(row[6]))
                task_z = int(float(row[7]))
                task_key = (task_x, task_z)

                if step < last_step and last_step >= 485:  # Step reset detected
                    if current_episode:
                        episodes[current_target].append(current_episode)
                    current_episode = []

                current_target = task_key
                current_episode.append({
                    "step": step,
                    "x": float(row[2]),
                    "z": float(row[3]),
                    "yaw": float(row[4]),
                    "reward": float(row[5]),
                    "task_x": task_x,
                    "task_z": task_z,
                })
                last_step = step
            except ValueError as e:
                print(f"Error parsing row {row}: {e}")
    if current_episode:
        episodes[current_target].append(current_episode)
    return episodes


def visualize_group(target_key, episodes, animation_speed=50, save_as_video=False):
    """
    Visualize a group of episodes with the same target as moving dots and show the target direction.
    """
    print(f"Visualizing group with target: {target_key}")
    episode_positions = [(list(map(lambda d: d["x"], ep)), list(map(lambda d: d["z"], ep))) for ep in episodes]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Target Direction: {target_key}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.grid(True)

    target_x, target_z = target_key
    magnitude = (target_x**2 + target_z**2)**0.5
    normalized_x, normalized_z = target_x / magnitude, target_z / magnitude
    arrow_length = max(10, magnitude * 10)
    ax.arrow(0, 0, normalized_x * arrow_length, normalized_z * arrow_length,
             head_width=2, head_length=3, fc='red', ec='red', label="Target Direction")
    agents = ax.scatter([], [], color="blue", label="Agent Position")

    all_x = [x for positions in episode_positions for x in positions[0]]
    all_z = [z for positions in episode_positions for z in positions[1]]
    margin = 10
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    ax.legend()

    def update(frame):
        x_coords, z_coords = [], []
        for x_positions, z_positions in episode_positions:
            if frame < len(x_positions):
                x_coords.append(x_positions[frame])
                z_coords.append(z_positions[frame])
        agents.set_offsets(list(zip(x_coords, z_coords)))
        return agents,

    ani = FuncAnimation(fig, update, frames=max(len(positions[0]) for positions in episode_positions), interval=animation_speed, blit=True)

    if save_as_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"target_{target_key[0]}_{target_key[1]}_{timestamp}.mp4")
        ani.save(output_path, writer="ffmpeg")
        print(f"Saved video to {output_path}")
    else:
        plt.show()


def visualize_samples_animated(episodes, sample_size, animation_speed=50, save_as_video=True):
    """
    Create an animated MP4 for 3 samples of episodes: start, middle, and end.
    """
    if len(episodes) < 3 * sample_size:
        print("Not enough episodes to create 3 samples.")
        return

    sample_start = episodes[:sample_size]
    sample_middle = episodes[len(episodes) // 2: len(episodes) // 2 + sample_size]
    sample_end = episodes[-sample_size:]

    colors = ['green', 'red', 'black']
    labels = ['1st gen', '2nd gen', '3rd gen']
    samples = [sample_start, sample_middle, sample_end]

    frames = max(max(len(ep) for ep in sample) for sample in samples)
    sample_positions = [
        [([data["x"] for data in episode], [data["z"] for data in episode]) for episode in sample]
        for sample in samples
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Animated Samples Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.grid(True)

    all_x = [data["x"] for sample in samples for episode in sample for data in episode]
    all_z = [data["z"] for sample in samples for episode in sample for data in episode]
    margin = 10
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    scatters = [ax.scatter([], [], color=color, label=label) for color, label in zip(colors, labels)]
    ax.legend()

    def update(frame):
        for scatter, sample_pos in zip(scatters, sample_positions):
            x_coords = []
            z_coords = []
            for x_positions, z_positions in sample_pos:
                if frame < len(x_positions):
                    x_coords.append(x_positions[frame])
                    z_coords.append(z_positions[frame])
            scatter.set_offsets(np.c_[x_coords, z_coords])
        return scatters

    ani = FuncAnimation(fig, update, frames=frames, interval=animation_speed, blit=True)

    if save_as_video:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"samples_visualization_{timestamp}.mp4")
        ani.save(output_path, writer="ffmpeg")
        print(f"Saved animation to {output_path}")
    else:
        plt.show()


def main():
    if not os.path.exists(LOG_DIR):
        print(f"Log directory '{LOG_DIR}' not found.")
        return

    episodes = read_all_log_files(LOG_DIR)

    print("\nAvailable Target Groups:")
    target_keys = list(episodes.keys())
    for i, key in enumerate(target_keys):
        print(f"{i}: Target {key}")

    try:
        choice = int(input("Enter the number of the target group to visualize: "))
        if 0 <= choice < len(target_keys):
            selected_target = target_keys[choice]
            selected_episodes = episodes[selected_target]
        else:
            print("Invalid choice. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return

    print("\nVisualization Modes:")
    print("1: Visualize single target group")
    print("2: Animated samples")
    try:
        mode = int(input("Enter the mode (1/2): "))
    except ValueError:
        print("Invalid input. Exiting.")
        return

    if mode == 1:
        animation_speed = int(input("Enter animation speed (default 50ms per frame): ") or "50")
        save_as_video = input("Save as video? (y/n): ").strip().lower() == "y"
        visualize_group(selected_target, selected_episodes, animation_speed, save_as_video)

    elif mode == 2:
        sample_size = int(input("Enter the sample size (e.g., 5): "))
        animation_speed = int(input("Enter animation speed (default 50ms per frame): ") or "50")
        visualize_samples_animated(selected_episodes, sample_size, animation_speed)
    else:
        print("Invalid mode selected. Exiting.")


if __name__ == "__main__":
    main()

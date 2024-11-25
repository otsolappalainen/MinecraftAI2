import csv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from datetime import datetime
import numpy as np

# Directory containing the environment log files
LOG_DIR = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen5"
OUTPUT_DIR = r"visualizations"  # Directory to save rendered videos or gifs


def is_valid_row(row):
    """
    Validate a row based on the given criteria.
    Focuses only on the first 10 fields and ignores extra fields.
    """
    required_fields = ['env_id', 'episode_id', 'step', 'x', 'z', 'yaw', 'pitch', 'reward', 'task_x', 'task_z']
    try:
        if len(row) < 10:
            print(f"Row has an insufficient number of fields: {len(row)}")
            return False

        # Extract required fields (only the first 10)
        row_dict = {header: value for header, value in zip(required_fields, row[:10])}

        # Validate 'step'
        step = int(float(row_dict['step']))
        if step < 0 or step > 500:
            print(f"Invalid step value: {step}")
            return False

        # Validate 'x', 'z', 'yaw', 'pitch', 'reward'
        float(row_dict['x'])      # x-coordinate
        float(row_dict['z'])      # z-coordinate
        float(row_dict['yaw'])    # yaw
        float(row_dict['pitch'])  # pitch
        float(row_dict['reward']) # reward

        # Validate 'task_x' and 'task_z'
        task_x = int(float(row_dict['task_x']))
        task_z = int(float(row_dict['task_z']))
        if task_x not in [-1, 0, 1] or task_z not in [-1, 0, 1]:
            print(f"Invalid task values: task_x={task_x}, task_z={task_z}")
            return False

        return True
    except (ValueError, KeyError) as e:
        print(f"Row validation failed: {row} ({e})")
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
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        episodes = read_single_log_file(file_path, episodes)

    # Print total episode count
    total_episodes = sum(len(eps) for eps in episodes.values())
    print(f"Total episodes processed: {total_episodes}")

    # Print episode count per target group
    for key, eps in episodes.items():
        print(f"Target {key}: {len(eps)} episodes")

    return episodes


def read_single_log_file(log_file, episodes):
    """
    Read a single log file and update the episodes dictionary.
    Detects episode boundaries based on step count drops or maximum episode size.
    """
    current_episode = []
    current_target = None
    last_step = -1
    row_count = 0

    with open(log_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            row_count += 1
            # Skip header row
            if row_count == 1 and row[0].lower() == "env_id":
                print(f"Skipping header row in file {log_file}.")
                continue
            if not is_valid_row(row):
                print(f"Invalid row skipped: {row}")
                continue
            try:
                # Extract required fields (only the first 10)
                row_dict = {header: value for header, value in zip(['env_id', 'episode_id', 'step', 'x', 'z', 'yaw', 'pitch', 'reward', 'task_x', 'task_z'], row[:10])}

                # Extract relevant data
                step = int(float(row_dict['step']))
                x = float(row_dict['x'])
                z = float(row_dict['z'])
                yaw = float(row_dict['yaw'])
                pitch = float(row_dict['pitch'])
                reward = float(row_dict['reward'])
                task_x = int(float(row_dict['task_x']))
                task_z = int(float(row_dict['task_z']))
                task_key = (task_x, task_z)

                # Detect episode boundary (step reset or step count limit)
                if last_step != -1 and (step < last_step or len(current_episode) >= 100):
                    print(f"New episode detected in file {log_file}: Step dropped from {last_step} to {step} or reached max steps.")
                    if current_episode:
                        episodes[current_target].append(current_episode)
                    current_episode = []  # Start a new episode

                # Store data for the current episode
                current_target = task_key
                current_episode.append({
                    "step": step,
                    "x": x,
                    "z": z,
                    "yaw": yaw,
                    "pitch": pitch,
                    "reward": reward,
                    "task_x": task_x,
                    "task_z": task_z,
                })
                last_step = step
            except ValueError as e:
                print(f"Error parsing row {row}: {e}")

    # Add the final episode to the target group
    if current_episode:
        episodes[current_target].append(current_episode)
        print(f"Final episode added for target {current_target} with {len(current_episode)} steps.")

    return episodes


def visualize_group(target_key, episodes, animation_speed=50, save_as_video=False):
    """
    Visualize a group of episodes with the same target as moving dots and show the target direction.
    Dynamically adjusts the plot limits to keep all agents in view.
    """
    print(f"Visualizing group with target: {target_key}")
    episode_positions = [(list(map(lambda d: d["x"], ep)), list(map(lambda d: d["z"], ep))) for ep in episodes]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Target Direction: {target_key}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.grid(True)

    target_x, target_z = target_key
    magnitude = (target_x**2 + target_z**2)**0.5 if (target_x**2 + target_z**2) > 0 else 1
    normalized_x, normalized_z = target_x / magnitude, target_z / magnitude
    arrow_length = max(10, magnitude * 10)
    ax.arrow(0, 0, normalized_x * arrow_length, normalized_z * arrow_length,
             head_width=2, head_length=3, fc='red', ec='red', label="Target Direction")
    agents = ax.scatter([], [], color="blue", label="Agent Position")

    ax.legend()

    def update(frame):
        x_coords, z_coords = [], []
        for x_positions, z_positions in episode_positions:
            if frame < len(x_positions):
                x_coords.append(x_positions[frame])
                z_coords.append(z_positions[frame])
        agents.set_offsets(list(zip(x_coords, z_coords)))

        # Dynamically adjust plot limits based on current positions
        if x_coords and z_coords:
            current_min_x, current_max_x = min(x_coords) - 5, max(x_coords) + 5
            current_min_z, current_max_z = min(z_coords) - 5, max(z_coords) + 5
            ax.set_xlim(current_min_x, current_max_x)
            ax.set_ylim(current_min_z, current_max_z)

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

    # Calculate overall limits
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
        print(f"{i}: Target {key} (Episodes: {len(episodes[key])})")

    if not target_keys:
        print("No valid target groups found. Exiting.")
        return

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
        try:
            animation_speed = int(input("Enter animation speed in ms per frame (default 50ms): ") or "50")
        except ValueError:
            animation_speed = 50
        save_as_video = input("Save as video? (y/n): ").strip().lower() == "y"
        visualize_group(selected_target, selected_episodes, animation_speed, save_as_video)

    elif mode == 2:
        try:
            sample_size = int(input("Enter the sample size (e.g., 5): "))
        except ValueError:
            print("Invalid sample size. Exiting.")
            return
        try:
            animation_speed = int(input("Enter animation speed in ms per frame (default 50ms): ") or "50")
        except ValueError:
            animation_speed = 50
        visualize_samples_animated(selected_episodes, sample_size, animation_speed, save_as_video=True)
    else:
        print("Invalid mode selected. Exiting.")


if __name__ == "__main__":
    main()

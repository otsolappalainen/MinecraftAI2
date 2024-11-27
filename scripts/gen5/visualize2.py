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


def is_valid_row(row_dict, required_fields):
    """
    Validate a row based on the given criteria.
    Focuses on specific fields by name.
    """
    try:
        # Check all required fields are present and not None or empty
        for field in required_fields:
            if field not in row_dict:
                print(f"Missing field: {field}")
                return False
            if row_dict[field] is None or row_dict[field].strip() == '':
                print(f"Empty value for field: {field}")
                return False

        # Validate 'step'
        step = int(float(row_dict['step']))
        if step < 0 or step > 500:
            print(f"Invalid step value: {step}")
            return False

        # Validate 'x', 'z', 'yaw', 'pitch', 'reward'
        x = float(row_dict['x'])
        z = float(row_dict['z'])
        yaw = float(row_dict['yaw'])
        pitch = float(row_dict['pitch'])
        reward = float(row_dict['reward'])

        # Validate 'task_x' and 'task_z'
        task_x = int(float(row_dict['task_x']))
        task_z = int(float(row_dict['task_z']))
        if task_x not in [-1, 0, 1] or task_z not in [-1, 0, 1]:
            print(f"Invalid task values: task_x={task_x}, task_z={task_z}")
            return False

        return True
    except (ValueError, TypeError) as e:
        print(f"Row validation failed: {row_dict} ({e})")
        return False


def read_all_log_files(log_dir):
    """
    Read and aggregate data from all matching log files in the directory.
    """
    episodes = defaultdict(list)
    total_skipped_rows = 0
    total_processed_rows = 0

    required_fields = ['env_id', 'episode_id', 'step', 'x', 'z', 'yaw', 'pitch', 'reward', 'task_x', 'task_z']

    for filename in os.listdir(log_dir):
        if not filename.startswith("training_data_env_") or not filename.endswith(".csv"):
            continue  # Skip files that don't match the pattern
        file_path = os.path.join(log_dir, filename)
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        processed, skipped = read_single_log_file(file_path, episodes, required_fields)
        total_processed_rows += processed
        total_skipped_rows += skipped

    # Print total episode count
    total_episodes = sum(len(eps) for eps in episodes.values())
    print(f"\nTotal episodes processed: {total_episodes}")
    print(f"Total rows processed: {total_processed_rows}")
    print(f"Total rows skipped: {total_skipped_rows}")

    # Print episode count per target group
    for key, eps in episodes.items():
        print(f"Target {key}: {len(eps)} episodes")

    return episodes


def read_single_log_file(log_file, episodes, required_fields):
    """
    Read a single log file and update the episodes dictionary.
    Detects episode boundaries based on step count drops or maximum episode size.
    Returns the number of processed and skipped rows.
    """
    current_episode = []
    current_target = None
    last_step = -1
    row_count = 0
    skipped_rows = 0
    processed_rows = 0

    with open(log_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if not is_valid_row(row, required_fields):
                print(f"Invalid row skipped: Row {row_count} in file {log_file}")
                skipped_rows += 1
                continue
            try:
                # Extract relevant data
                step = int(float(row['step']))
                x = float(row['x'])
                z = float(row['z'])
                yaw = float(row['yaw'])
                pitch = float(row['pitch'])
                reward = float(row['reward'])
                task_x = int(float(row['task_x']))
                task_z = int(float(row['task_z']))
                task_key = (task_x, task_z)

                # Detect episode boundary (step reset or step count limit)
                if last_step != -1 and (step < last_step or len(current_episode) >= 500):
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
                processed_rows += 1
            except (ValueError, TypeError) as e:
                print(f"Error parsing row {row_count} in file {log_file}: {e}")
                skipped_rows += 1
                continue

    # Add the final episode to the target group
    if current_episode:
        episodes[current_target].append(current_episode)
        print(f"Final episode added for target {current_target} with {len(current_episode)} steps.")

    return processed_rows, skipped_rows


def visualize_group(target_key, episodes, animation_speed=50, save_as_video=False):
    """
    Visualize a group of episodes with the same target as moving dots and show the target direction.
    Dynamically adjusts the plot limits to keep all agents in view.
    """
    print(f"\nVisualizing group with target: {target_key}")
    episode_positions = [(ep["x"], ep["z"]) for ep in episodes]

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
    agents, = ax.plot([], [], 'bo', markersize=2, label="Agent Position")

    ax.legend()

    def update(frame):
        x_coords, z_coords = [], []
        for ep in episodes:
            if frame < len(ep):
                x_coords.append(ep[frame][0])
                z_coords.append(ep[frame][1])
        agents.set_data(x_coords, z_coords)

        # Dynamically adjust plot limits based on current positions
        if x_coords and z_coords:
            current_min_x, current_max_x = min(x_coords) - 5, max(x_coords) + 5
            current_min_z, current_max_z = min(z_coords) - 5, max(z_coords) + 5
            ax.set_xlim(current_min_x, current_max_x)
            ax.set_ylim(current_min_z, current_max_z)

        return agents,

    # Determine the maximum number of steps in episodes
    max_steps = max(len(ep) for ep in episodes) if episodes else 0
    ani = FuncAnimation(fig, update, frames=max_steps, interval=animation_speed, blit=True)

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

    # Determine the maximum number of steps across all samples
    frames = max(len(ep) for sample in samples for ep in sample) if samples else 0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Animated Samples Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Z Position")
    ax.grid(True)

    # Calculate overall limits
    all_x = [step["x"] for sample in samples for ep in sample for step in ep]
    all_z = [step["z"] for sample in samples for ep in sample for step in ep]
    margin = 10
    if all_x and all_z:
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_z) - margin, max(all_z) + margin)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    scatters = [ax.scatter([], [], color=color, label=label) for color, label in zip(colors, labels)]
    ax.legend()

    def update(frame):
        for scatter, sample in zip(scatters, samples):
            x_coords = []
            z_coords = []
            for episode in sample:
                if frame < len(episode):
                    x_coords.append(episode[frame]["x"])
                    z_coords.append(episode[frame]["z"])
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

    episodes_dict = read_all_log_files(LOG_DIR)

    # Flatten the episodes_dict into a list for visualization
    all_target_keys = list(episodes_dict.keys())
    all_episodes = []
    for target_key in all_target_keys:
        all_episodes.extend(episodes_dict[target_key])

    if not all_episodes:
        print("No valid episodes found. Exiting.")
        return

    print("\nAvailable Target Groups:")
    target_keys = list(episodes_dict.keys())
    for i, key in enumerate(target_keys):
        print(f"{i}: Target {key} (Episodes: {len(episodes_dict[key])} episodes)")

    if not target_keys:
        print("No valid target groups found. Exiting.")
        return

    try:
        choice = int(input("\nEnter the number of the target group to visualize: "))
        if 0 <= choice < len(target_keys):
            selected_target = target_keys[choice]
            selected_episodes = episodes_dict[selected_target]
        else:
            print("Invalid choice. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return

    if not selected_episodes:
        print(f"No episodes found for target {selected_target}. Exiting.")
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
            if sample_size <= 0:
                print("Sample size must be positive. Exiting.")
                return
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

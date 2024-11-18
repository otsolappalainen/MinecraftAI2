import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

TRAJECTORY_LOG_PATH = r"E:\model_spam\trajectories"  # Path where trajectories are saved


def load_all_trajectories():
    """
    Load all trajectory files from the folder.
    """
    trajectory_files = [f for f in os.listdir(TRAJECTORY_LOG_PATH) if f.endswith(".pkl")]
    if not trajectory_files:
        print("No trajectory files found.")
        return None

    trajectories = []
    for file_name in trajectory_files:
        file_path = os.path.join(TRAJECTORY_LOG_PATH, file_name)
        with open(file_path, "rb") as file:
            trajectory = pickle.load(file)
            trajectories.append((file_name, trajectory))
    return trajectories


def animate_trajectory(trajectory, file_name, task_numbers):
    """
    Create an animation of the trajectory using matplotlib.
    """
    x_positions = []
    z_positions = []
    rewards = []

    # Extract x, z positions and rewards
    for step, (obs, action, reward) in enumerate(trajectory):
        obs = np.array(obs).flatten()  # Ensure the observation is flattened
        if len(obs) < 25:  # Validate observation size
            print(f"Skipping step {step} in {file_name}: Observation size mismatch.")
            continue

        x = obs[-25]  # Adjust this index to match where x is stored
        z = obs[-24]  # Adjust this index to match where z is stored
        x_positions.append(float(x))  # Ensure the value is a float
        z_positions.append(float(z))  # Ensure the value is a float
        rewards.append(float(reward))  # Ensure reward is a float

    total_reward = sum(rewards)

    # Ensure data integrity
    if len(x_positions) == 0 or len(z_positions) == 0:
        print(f"Skipping animation for {file_name}: No valid positions.")
        return

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Trajectory: {file_name}\nTask: {task_numbers}, Total Reward: {total_reward:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)

    # Determine plot limits with a margin
    margin = 10
    ax.set_xlim(min(x_positions) - margin, max(x_positions) + margin)
    ax.set_ylim(min(z_positions) - margin, max(z_positions) + margin)

    # Initialize plot elements
    path, = ax.plot([], [], label="Agent Path", lw=2)
    current_position, = ax.plot([], [], "ro", label="Current Position")

    def update(frame):
        # Ensure frame index is valid
        if frame >= len(x_positions):
            return path, current_position

        # Update the path and current position
        path.set_data(x_positions[:frame + 1], z_positions[:frame + 1])
        current_position.set_data([x_positions[frame]], [z_positions[frame]])  # Ensure sequence format
        return path, current_position

    ani = FuncAnimation(fig, update, frames=len(x_positions), interval=100, blit=True)

    # Display the animation
    plt.legend()
    plt.show(block=True)
    plt.close(fig)



def render_trajectories(trajectories):
    """
    Automatically render all trajectories using matplotlib animations.
    """
    for file_name, trajectory in trajectories:
        print(f"Rendering trajectory from file: {file_name}")

        # Extract task information (last 20 numbers in observation)
        obs = trajectory[0][0][0]  # First observation
        task_numbers = obs[-20:-18].astype(int)  # First two numbers of the task array

        # Animate the trajectory
        animate_trajectory(trajectory, file_name, task_numbers)


def main():
    print("Loading all trajectories...")
    trajectories = load_all_trajectories()
    if trajectories is None:
        return

    print("Rendering all trajectories...")
    render_trajectories(trajectories)
    print("Rendering completed for all trajectories.")


if __name__ == "__main__":
    main()
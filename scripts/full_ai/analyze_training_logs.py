import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Define the folder containing log files
log_folder = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\training_logs"

def parse_log_file(file_path):
    """
    Parse a single log file to extract training data.
    """
    data = {
        "Step": [],
        "X": [],
        "Y": [],
        "Z": [],
        "Yaw": [],
        "Pitch": [],
        "Action": [],
        "Reward": [],
        "Cumulative Reward": []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(
                r"Step (\d+): X = ([\d\.-]+), Y = ([\d\.-]+), Z = ([\d\.-]+), "
                r"Yaw = ([\d\.-]+), Pitch = ([\d\.-]+), Action = (\d+), "
                r"Reward = ([\d\.-]+), Cumulative Reward = ([\d\.-]+)", line
            )
            if match:
                step, x, y, z, yaw, pitch, action, reward, cum_reward = match.groups()
                data["Step"].append(int(step))
                data["X"].append(float(x))
                data["Y"].append(float(y))
                data["Z"].append(float(z))
                data["Yaw"].append(float(yaw))
                data["Pitch"].append(float(pitch))
                data["Action"].append(int(action))
                data["Reward"].append(float(reward))
                data["Cumulative Reward"].append(float(cum_reward))
    
    return pd.DataFrame(data)

def plot_graphs(df, file_name, output_dir):
    """
    Plot graphs from a dataframe and save them as images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Reward over Steps
    plt.figure()
    plt.plot(df["Step"], df["Reward"], label="Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Steps - {file_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name}_reward.png"))
    plt.close()

    # Plot Cumulative Reward over Steps
    plt.figure()
    plt.plot(df["Step"], df["Cumulative Reward"], label="Cumulative Reward", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Cumulative Reward Over Steps - {file_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name}_cumulative_reward.png"))
    plt.close()

    # Plot Position (X, Z) over Steps
    plt.figure()
    plt.plot(df["Step"], df["X"], label="X Position")
    plt.plot(df["Step"], df["Z"], label="Z Position", color="green")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title(f"Position Over Steps - {file_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name}_position.png"))
    plt.close()

    # Plot Yaw over Steps
    plt.figure()
    plt.plot(df["Step"], df["Yaw"], label="Yaw", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Yaw")
    plt.title(f"Yaw Over Steps - {file_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name}_yaw.png"))
    plt.close()

    # Plot Average Reward per Action
    action_rewards = df.groupby("Action")["Reward"].mean().sort_index()
    plt.figure()
    plt.bar(action_rewards.index, action_rewards.values)
    plt.xlabel("Action")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward per Action - {file_name}")
    plt.xticks(action_rewards.index)
    plt.savefig(os.path.join(output_dir, f"{file_name}_avg_reward_per_action.png"))
    plt.close()

def process_all_logs(log_folder, output_dir="graphs"):
    """
    Process all log files in the folder and generate graphs.
    """
    for file_name in os.listdir(log_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(log_folder, file_name)
            print(f"Processing {file_name}...")
            df = parse_log_file(file_path)
            plot_graphs(df, os.path.splitext(file_name)[0], output_dir)

# Run the script
process_all_logs(log_folder)
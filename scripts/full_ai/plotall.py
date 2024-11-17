import os
import re
import pandas as pd

# Define the folder containing log files
log_folder = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\training_logs"

# Define the output file
output_csv = "combined_data_corrected.csv"

def parse_log_file(file_path):
    """
    Parse a single log file to extract training data.
    """
    data = {
        "Episode": [],  # To track episodes explicitly
        "Step": [],
        "X": [],
        "Y": [],
        "Z": [],
        "Yaw": [],
        "Pitch": [],
        "Action": [],
        "Action Name": [],
        "Reward": [],
        "Cumulative Reward": []
    }
    
    episode = os.path.basename(file_path).split('_')[1]  # Extract episode ID from file name

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(
                r"Step (\d+): X = ([\d\.-]+), Y = ([\d\.-]+), Z = ([\d\.-]+), "
                r"Yaw = ([\d\.-]+), Pitch = ([\d\.-]+), Action = (\d+), "
                r"Action Name = (\w+), Reward = ([\d\.-]+), Cumulative Reward = ([\d\.-]+)", line
            )
            if match:
                step, x, y, z, yaw, pitch, action, action_name, reward, cum_reward = match.groups()
                data["Episode"].append(int(episode))  # Add episode identifier
                data["Step"].append(int(step))
                data["X"].append(float(x))
                data["Y"].append(float(y))
                data["Z"].append(float(z))
                data["Yaw"].append(float(yaw))
                data["Pitch"].append(float(pitch))
                data["Action"].append(int(action))
                data["Action Name"].append(action_name)
                data["Reward"].append(float(reward))
                data["Cumulative Reward"].append(float(cum_reward))
    
    print(f"Parsed {len(data['Step'])} steps from {file_path}")
    return pd.DataFrame(data)

def combine_log_files(log_folder, output_csv):
    """
    Combine all log files into a single CSV file, ensuring proper order.
    """
    all_data = []
    for file_name in os.listdir(log_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(log_folder, file_name)
            print(f"Processing {file_name}...")
            df = parse_log_file(file_path)
            all_data.append(df)
    
    # Combine all data into one DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by Episode and Step to ensure sequential order
    combined_df = combined_df.sort_values(by=["Episode", "Step"]).reset_index(drop=True)
    print(f"Combined DataFrame contains {len(combined_df)} steps.")
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined data saved to {output_csv}")

# Run the function to process and combine logs
combine_log_files(log_folder, output_csv)
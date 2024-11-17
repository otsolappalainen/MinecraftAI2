import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Define the folder containing log files
log_folder = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\training_logs"

# Parameters for graph resolution
BIN_SIZE = 1  # Number of steps per bin for aggregation (e.g., 10, 100, 1000)

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
        "Action Name": [],
        "Reward": [],
        "Cumulative Reward": [],
    }

    with open(file_path, "r") as file:
        for line in file:
            match = re.match(
                r"Step (\d+): X = ([\d\.-]+), Y = ([\d\.-]+), Z = ([\d\.-]+), "
                r"Yaw = ([\d\.-]+), Pitch = ([\d\.-]+), Action = (\d+), "
                r"Action Name = (\w+), Reward = ([\d\.-]+), Cumulative Reward = ([\d\.-]+)",
                line,
            )
            if match:
                step, x, y, z, yaw, pitch, action, action_name, reward, cum_reward = match.groups()
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


def aggregate_data(df, bin_size):
    """
    Aggregate data into bins of size `bin_size` to reduce resolution.
    """
    df["Bin"] = df["Step"] // bin_size
    # Aggregate only numeric columns
    numeric_columns = ["Step", "X", "Y", "Z", "Yaw", "Pitch", "Reward", "Cumulative Reward"]
    aggregated = df[numeric_columns].groupby(df["Bin"]).mean().reset_index()
    aggregated["Step"] = aggregated["Bin"] * bin_size  # Map bins back to steps
    return aggregated


def plot_combined_graphs(df, pdf, bin_size):
    """
    Plot combined graphs from the aggregated dataframe and save to a PDF.
    """
    # Aggregate data for reduced resolution
    aggregated_df = aggregate_data(df, bin_size)

    # Plot Reward over Steps
    plt.figure()
    plt.plot(aggregated_df["Step"], aggregated_df["Reward"], label="Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Steps (Bin Size: {bin_size})")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot Cumulative Reward over Steps
    plt.figure()
    plt.plot(aggregated_df["Step"], aggregated_df["Cumulative Reward"], label="Cumulative Reward", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Cumulative Reward Over Steps (Bin Size: {bin_size})")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot Position (X, Z) over Steps
    plt.figure()
    plt.plot(aggregated_df["Step"], aggregated_df["X"], label="X Position")
    plt.plot(aggregated_df["Step"], aggregated_df["Z"], label="Z Position", color="green")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title(f"Position Over Steps (Bin Size: {bin_size})")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot Yaw over Steps
    plt.figure()
    plt.plot(aggregated_df["Step"], aggregated_df["Yaw"], label="Yaw", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Yaw")
    plt.title(f"Yaw Over Steps (Bin Size: {bin_size})")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot Average Reward per Action
    action_rewards = df.groupby("Action")["Reward"].mean().sort_index()
    plt.figure()
    plt.bar(action_rewards.index, action_rewards.values)
    plt.xlabel("Action")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Action")
    plt.xticks(action_rewards.index)
    pdf.savefig()  # Save the plot to the PDF
    plt.close()


def plot_every_100th_datapoint(df, pdf):
    """
    Plot a graph using every 100th datapoint from the DataFrame and save to a PDF.
    """
    # Select every 100th row
    every_100th_df = df.iloc[::100, :]

    # Plot every 100th Reward over Steps
    plt.figure()
    plt.plot(every_100th_df["Step"], every_100th_df["Reward"], label="Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Over Steps (Every 100th Datapoint)")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot every 100th Cumulative Reward over Steps
    plt.figure()
    plt.plot(every_100th_df["Step"], every_100th_df["Cumulative Reward"], label="Cumulative Reward", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Steps (Every 100th Datapoint)")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot Position (X, Z) for every 100th datapoint
    plt.figure()
    plt.plot(every_100th_df["Step"], every_100th_df["X"], label="X Position")
    plt.plot(every_100th_df["Step"], every_100th_df["Z"], label="Z Position", color="green")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Position Over Steps (Every 100th Datapoint)")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()

    # Plot Yaw over Steps for every 100th datapoint
    plt.figure()
    plt.plot(every_100th_df["Step"], every_100th_df["Yaw"], label="Yaw", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Yaw")
    plt.title("Yaw Over Steps (Every 100th Datapoint)")
    plt.legend()
    pdf.savefig()  # Save the plot to the PDF
    plt.close()


def process_and_plot_all_logs(log_folder, output_pdf="combined_graphs.pdf", output_csv="combined_data.csv", bin_size=BIN_SIZE):
    """
    Process all log files, combine data, generate graphs in a single PDF, and save raw data to a CSV.
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
    combined_df = combined_df.sort_values("Step").reset_index(drop=True)
    print(f"Combined DataFrame contains {len(combined_df)} steps.")

    # Save the raw data to a CSV file
    combined_df.to_csv(output_csv, index=False)
    print(f"Raw data saved to {output_csv}")

    # Open a multi-page PDF for all plots
    with PdfPages(output_pdf) as pdf:
        # Plot combined graphs
        plot_combined_graphs(combined_df, pdf, bin_size)
        # Plot every 100th datapoint graphs
        plot_every_100th_datapoint(combined_df, pdf)

    print(f"Graphs saved to {output_pdf}")


# Run the script
process_and_plot_all_logs(log_folder)
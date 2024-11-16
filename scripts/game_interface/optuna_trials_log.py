import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

header = ["Trial", "Value (Reward)", "n_steps", "batch_size", "learning_rate", "n_epochs"]

# Load the trial log data
def load_optuna_log(log_file="optuna_trials_log.csv"):
    data = pd.read_csv(log_file)
    return data

# Function to plot rewards over trials
def plot_trial_rewards(data):
    plt.figure(figsize=(12, 8))
    plt.plot(data["Trial"], data["Value (Reward)"], marker='o', label="Reward per Trial")
    plt.xlabel("Trial")
    plt.ylabel("Reward")
    plt.title("Reward Progression Across Trials")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot hyperparameter importances
def plot_hyperparameter_importances(data):
    sns.set(style="whitegrid")

    # Convert columns to numeric where possible (for sorting and scaling)
    data_numeric = data.copy()
    for col in ["n_steps", "batch_size", "learning_rate", "n_epochs"]:
        data_numeric[col] = pd.to_numeric(data[col], errors="coerce")

    # Calculate correlation of hyperparameters with reward
    corr = data_numeric[["Value (Reward)", "n_steps", "batch_size", "learning_rate", "n_epochs"]].corr()
    corr_target = corr["Value (Reward)"].drop("Value (Reward)").abs()

    # Sort features by correlation with reward
    sorted_features = corr_target.sort_values(ascending=False)
    sorted_features.plot(kind="bar", color="skyblue")
    plt.title("Hyperparameter Importance (Correlation with Reward)")
    plt.xlabel("Hyperparameters")
    plt.ylabel("Absolute Correlation with Reward")
    plt.show()

def plot_hyperparameter_importances2(data):
    sns.set(style="whitegrid")

    # Convert columns to numeric where possible (for sorting and scaling)
    data_numeric = data.copy()
    for col in ["n_steps", "batch_size", "learning_rate", "n_epochs"]:
        data_numeric[col] = pd.to_numeric(data[col], errors="coerce")

    # Calculate correlation of hyperparameters with reward
    corr = data_numeric[["Value (Reward)", "n_steps", "batch_size", "learning_rate", "n_epochs"]].corr()
    corr_target = corr["Value (Reward)"].drop("Value (Reward)")

    # Plot correlation (including direction) for each hyperparameter
    plt.figure(figsize=(8, 5))
    corr_target.plot(kind="bar", color=["skyblue" if c > 0 else "salmon" for c in corr_target])
    plt.title("Hyperparameter Importance (Correlation with Reward, Including Direction)")
    plt.xlabel("Hyperparameters")
    plt.ylabel("Correlation with Reward")
    plt.axhline(0, color='gray', linestyle='--')
    plt.show()

    # Plot scatter plots for each hyperparameter against reward
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("Reward vs Hyperparameters")
    
    for i, col in enumerate(["n_steps", "batch_size", "learning_rate", "n_epochs"]):
        sns.scatterplot(data=data_numeric, x=col, y="Value (Reward)", ax=axes[i])
        sns.lineplot(data=data_numeric, x=col, y="Value (Reward)", ax=axes[i], color="orange", ci=None)
        axes[i].set_title(f"Reward vs {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Reward")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle
    plt.show()


def ensure_header(file_path="optuna_trials_log.csv"):
    header = ["Trial", "Value (Reward)", "n_steps", "batch_size", "learning_rate", "n_epochs"]
    
    try:
        # Check if the file exists and has a header
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            # If the file has the correct header, do nothing and exit the function
            if first_line == ",".join(header):
                return

            # Otherwise, read the rest of the file contents
            data = f.readlines()
        
        # Re-write the file with the header and the existing data
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write the header
            # Write the existing data only if it was previously read
            if data:
                for line in data:
                    writer.writerow(line.strip().split(','))  # Write each line as CSV
            print("Header added to the file.")

    except FileNotFoundError:
        # If the file does not exist, create it with only the header
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            print("File created with header.")

    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    
    ensure_header("C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\optuna_trials_log.csv")

    # Load the main Optuna trial data
    data = load_optuna_log("optuna_trials_log.csv")
    
    # Plot the reward progression across trials
    plot_trial_rewards(data)

    # Plot hyperparameter importances
    plot_hyperparameter_importances(data)
    plot_hyperparameter_importances2(data)


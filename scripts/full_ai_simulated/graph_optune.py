import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = r"E:\CNN\optuna_trials_log.csv"  # Replace with the correct path
data = pd.read_csv(file_path)

# Define parameters for analysis
parameters = ["learning_rate", "buffer_size", "batch_size", "gamma", "exploration_fraction", "exploration_final_eps"]
mean_reward = "mean_reward"

# Define bin sizes for histograms
bin_sizes = {
    "learning_rate": 6,
    "buffer_size": 5,
    "batch_size": 4,
    "gamma": 6,
    "exploration_fraction": 5,
    "exploration_final_eps": 5,
}

# Analyze each parameter and plot results
for param in parameters:
    plt.figure(figsize=(10, 6))
    
    # Determine bins based on the range of the parameter
    if param in bin_sizes:
        bins = np.linspace(data[param].min(), data[param].max(), bin_sizes[param])
    else:
        bins = 10  # Default bin count
    
    # Plot histogram
    plt.hist(data[param], bins=bins, weights=data[mean_reward], alpha=0.6, color='blue', label="Weighted by Mean Reward")
    plt.hist(data[param], bins=bins, alpha=0.6, color='orange', label="Count")

    # Add labels and title
    plt.title(f"Parameter: {param} vs Mean Reward")
    plt.xlabel(param)
    plt.ylabel("Mean Reward / Count")
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Estimate good parameter ranges based on mean reward
summary = {}
for param in parameters:
    grouped = data.groupby(pd.cut(data[param], bins=bin_sizes.get(param, 10)))
    summary[param] = grouped[mean_reward].mean()


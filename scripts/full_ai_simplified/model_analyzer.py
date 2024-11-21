import os
import re
import zipfile
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from tqdm import tqdm  # For progress bars

# Ensure consistent style for plots
sns.set(style="whitegrid")

def extract_timestep_from_filename(filename):
    """
    Extracts the timestep integer from a filename.
    Expected filename format: 'model_step_<timestep>.zip'.
    Returns None if the filename does not match the expected format.
    """
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        # If the filename contains a timestamp instead of a timestep, return None
        if re.search(r'\d{14}', filename):  # Matches a 14-digit timestamp like 20241121193228
            return None
        return None

def extract_state_dict(zip_path):
    """
    Extracts the state_dict from a model .zip file.
    Adapts to Stable-Baselines3 structure using 'policy.pth'.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        # Look for 'policy.pth' for model weights
        if 'policy.pth' in file_list:
            zip_ref.extract('policy.pth', path='temp_extracted')
            state_dict = torch.load(os.path.join('temp_extracted', 'policy.pth'), map_location='cpu')
            # Clean up the extracted file
            os.remove(os.path.join('temp_extracted', 'policy.pth'))
            os.rmdir('temp_extracted')
            return state_dict
        else:
            print(f"'policy.pth' not found in {zip_path}. Contents: {file_list}")
            return None


def analyze_weights(state_dict):
    """
    Analyzes the weights in the state_dict.
    Returns a dictionary with layer names as keys and their weight statistics as values.
    """
    layer_stats = {}
    for layer_name, weights in state_dict.items():
        # Only analyze weight tensors, skip biases or other parameters
        if 'weight' in layer_name:
            weights_tensor = weights.cpu().numpy()
            stats = {
                'mean': weights_tensor.mean(),
                'std': weights_tensor.std(),
                'min': weights_tensor.min(),
                'max': weights_tensor.max()
            }
            layer_stats[layer_name] = stats
    return layer_stats

def aggregate_stats(model_dir):
    """
    Iterates through all .zip files in the directory, extracts and analyzes weights.
    Returns a pandas DataFrame with aggregated statistics.
    """
    data = defaultdict(list)
    timesteps = []

    # List and sort all .zip files based on timestep
    zip_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    zip_files = sorted(zip_files, key=lambda x: extract_timestep_from_filename(x) or float('inf'))

    print("Processing model files...")
    for zip_file in tqdm(zip_files):
        timestep = extract_timestep_from_filename(zip_file)
        if timestep is None:
            print(f"Skipping non-timestep file: {zip_file}")
            continue
        timesteps.append(timestep)
        zip_path = os.path.join(model_dir, zip_file)
        state_dict = extract_state_dict(zip_path)
        if state_dict is None:
            continue
        layer_stats = analyze_weights(state_dict)
        for layer, stats in layer_stats.items():
            data['timestep'].append(timestep)
            data['layer'].append(layer)
            data['mean'].append(stats['mean'])
            data['std'].append(stats['std'])
            data['min'].append(stats['min'])
            data['max'].append(stats['max'])
    
    df = pd.DataFrame(data)
    return df

def plot_statistics(df, output_dir):
    """
    Creates and saves plots for weight statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot Mean Weights Over Time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='timestep', y='mean', hue='layer')
    plt.title('Mean of Weights Over Training Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Mean Weight Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_weights_over_time.png'))
    plt.close()

    # Plot Std of Weights Over Time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='timestep', y='std', hue='layer')
    plt.title('Standard Deviation of Weights Over Training Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Std of Weights')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'std_weights_over_time.png'))
    plt.close()

    # Plot Min and Max Weights Over Time for each layer
    layers = df['layer'].unique()
    for layer in layers:
        layer_df = df[df['layer'] == layer]
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=layer_df, x='timestep', y='min', label='Min')
        sns.lineplot(data=layer_df, x='timestep', y='max', label='Max')
        plt.title(f'Min and Max Weights Over Time for {layer}')
        plt.xlabel('Timestep')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.tight_layout()
        # Replace slashes in layer names to avoid file path issues
        safe_layer_name = layer.replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'min_max_weights_{safe_layer_name}.png'))
        plt.close()

    # Optionally, save the DataFrame for further analysis
    df.to_csv(os.path.join(output_dir, 'weight_statistics.csv'), index=False)
    print(f"Plots saved in {output_dir}")

def main():
    # Directory containing the model .zip files
    model_dir = 'models'  # Change this if your models are in a different directory
    output_dir = 'weight_analysis_plots'

    # Aggregate statistics from all models
    df = aggregate_stats(model_dir)

    if df.empty:
        print("No data to plot. Ensure that the model .zip files contain 'pytorch_model.bin'.")
        return

    # Sort DataFrame by timestep and layer for consistent plotting
    df = df.sort_values(by=['timestep', 'layer'])

    # Plot the statistics
    plot_statistics(df, output_dir)

if __name__ == "__main__":
    main()

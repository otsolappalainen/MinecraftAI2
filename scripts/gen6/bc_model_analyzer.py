import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def collect_model_files(directory):
    """
    Collect all the model files (both epoch models and final model) in the specified directory.
    """
    model_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            model_files.append(os.path.join(directory, filename))
    return sorted(model_files)  # Sort by epoch order if files are named consistently

def load_model(file_path):
    """
    Load the model from the given file.
    Assumes the model is a PyTorch model.
    """
    return torch.load(file_path)

def compare_weights(model_files):
    """
    Compare the weights across multiple model files and plot the changes.
    """
    weight_differences = []  # To store the L2 norm of weight differences

    # Load the first model
    previous_model = load_model(model_files[0])
    
    # Collect initial weights
    previous_weights = {name: param.detach().cpu().numpy() for name, param in previous_model.items()}
    
    for i, model_file in enumerate(model_files[1:], start=1):
        current_model = load_model(model_file)
        
        # Collect current weights
        current_weights = {name: param.detach().cpu().numpy() for name, param in current_model.items()}
        
        # Compute the L2 norm of weight differences for each parameter
        weight_diff = []
        for name in current_weights:
            if name in previous_weights:
                diff = np.linalg.norm(current_weights[name] - previous_weights[name])
                weight_diff.append(diff)
        
        # Store the total weight difference for this epoch
        weight_differences.append(np.sum(weight_diff))
        
        # Update previous_weights to be the current model's weights
        previous_weights = current_weights
    
    # Add the final model's weight change (from the last epoch)
    final_model = load_model(model_files[-1])
    final_weights = {name: param.detach().cpu().numpy() for name, param in final_model.items()}
    
    weight_diff = []
    for name in final_weights:
        if name in previous_weights:
            diff = np.linalg.norm(final_weights[name] - previous_weights[name])
            weight_diff.append(diff)
    
    weight_differences.append(np.sum(weight_diff))

    return weight_differences

def check_parameter_changes(model_files, sample_size=10000):
    """
    Check how many values in the model are the same across epochs. It compares a random sample of parameters
    across each epoch and calculates the percentage of values that are different.
    """
    change_percentages = []
    
    # Load the first model
    previous_model = load_model(model_files[0])
    previous_weights = {name: param.detach().cpu().numpy() for name, param in previous_model.items()}

    for model_file in model_files[1:]:
        current_model = load_model(model_file)
        current_weights = {name: param.detach().cpu().numpy() for name, param in current_model.items()}
        
        # Randomly sample up to `sample_size` values across the weights
        sampled_differences = []
        for name in current_weights:
            if name in previous_weights:
                previous_weight_values = previous_weights[name].flatten()
                current_weight_values = current_weights[name].flatten()
                
                # Get a random sample from both weight arrays
                indices = np.random.choice(len(previous_weight_values), min(sample_size, len(previous_weight_values)), replace=False)
                previous_sampled_values = previous_weight_values[indices]
                current_sampled_values = current_weight_values[indices]
                
                # Count how many values are the same
                same_count = np.sum(previous_sampled_values == current_sampled_values)
                total_sampled = len(indices)
                
                # Calculate percentage of values that are the same
                sampled_differences.append(1 - same_count / total_sampled)
        
        # Calculate average change percentage for this model
        if sampled_differences:
            change_percentages.append(np.mean(sampled_differences) * 100)
        else:
            change_percentages.append(0)

        # Update previous model for next comparison
        previous_weights = current_weights
    
    return change_percentages

def plot_weight_changes(weight_differences, model_files):
    """
    Plot the weight changes across epochs.
    """
    epochs = [os.path.basename(file).split('_')[-1].split('.')[0] for file in model_files]  # Extract epoch numbers from filenames
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, weight_differences, marker='o')
    plt.title('Weight Changes Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Norm of Weight Changes')
    plt.grid(True)
    plt.show()

def plot_parameter_changes(change_percentages, model_files):
    """
    Plot the percentage of parameter changes across epochs.
    """
    epochs = [os.path.basename(file).split('_')[-1].split('.')[0] for file in model_files]  # Extract epoch numbers from filenames
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[1:], change_percentages, marker='o', color='red')
    plt.title('Percentage of Parameter Differences Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('% of Parameters Changed')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the folder containing your model files
    model_folder = "C:/Users/odezz/source/MinecraftAI2/scripts/gen6/models_bc"
    
    # Collect all model files in the folder
    model_files = collect_model_files(model_folder)
    
    # Compare the weights between epochs
    weight_differences = compare_weights(model_files)
    
    # Plot the changes in weights
    plot_weight_changes(weight_differences, model_files)
    
    # Check the percentage of parameter changes
    change_percentages = check_parameter_changes(model_files)
    
    # Plot the percentage of changes
    plot_parameter_changes(change_percentages, model_files)

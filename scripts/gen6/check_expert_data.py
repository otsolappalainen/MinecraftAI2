import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_and_visualize_data(pkl_file, num_samples=5):
    """
    Load and visualize data from a .pkl file.
    
    Parameters:
        pkl_file (str): Path to the .pkl file.
        num_samples (int): Number of samples to display.
    """
    try:
        # Load the data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Check the type and length of the data
        if not isinstance(data, list):
            print(f"Expected a list of samples, got {type(data)}.")
            return
        
        print(f"Loaded {len(data)} samples from {pkl_file}.")
        
        # Display a few samples
        for idx, sample in enumerate(data[:num_samples]):
            print(f"\n--- Sample {idx + 1} ---")
            action = sample.get('action', None)
            observation = sample.get('observation', None)
            timestamp = sample.get('timestamp', None)
            
            print(f"Action: {action}")
            print(f"Timestamp: {timestamp}")
            
            if observation:
                # Display image data
                if 'image' in observation:
                    img = observation['image']
                    if isinstance(img, np.ndarray) and img.shape[0] == 3:
                        plt.figure(figsize=(4, 4))
                        plt.imshow(np.transpose(img, (1, 2, 0)))  # Transpose for HWC format
                        plt.title(f"Sample {idx + 1} - Action: {action}")
                        plt.axis('off')
                        plt.show()
                    else:
                        print(f"Invalid image shape: {img.shape}")
                
                # Display scalar metadata
                if 'other' in observation:
                    other = observation['other']
                    print(f"Other Metadata (first 10 values): {other[:10]}")
            else:
                print("Observation missing in this sample.")

        print("\nVisualization complete.")
    
    except Exception as e:
        print(f"Error reading .pkl file: {e}")

# Specify the path to your .pkl file
pkl_file_path = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\expert_data\session_20241128_213941\expert_data.pkl"  # Update this with your file path
load_and_visualize_data(pkl_file_path, num_samples=5)
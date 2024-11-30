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
            print(f"Action: {sample}")
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
                
                # Display scalar metadata with labels
                if 'other' in observation:
                    other = observation['other']
                    print(f"observation data: {observation}")
                    if len(other) >= 8:
                        # Assuming the first 8 values correspond to known metadata
                        x = other[0]
                        z = other[1]
                        y = other[2]
                        sin_yaw = other[3]
                        cos_yaw = other[4]
                        health = other[5]
                        hunger = other[6]
                        alive = other[7]

                        print(f"Scalar Metadata:")
                        print(f"x: {x:.2f}")
                        print(f"z: {z:.2f}")
                        print(f"y: {y:.2f}")
                        print(f"sin(yaw): {sin_yaw:.2f}")
                        print(f"cos(yaw): {cos_yaw:.2f}")
                        print(f"Health: {health:.2f}")
                        print(f"Hunger: {hunger:.2f}")
                        print(f"Alive: {alive}")
                    else:
                        print("Insufficient 'other' metadata for labeling.")
                
                # Display task array if available
                if 'task' in observation:
                    task = observation['task']
                    print(f"Task Array: {task}")
                else:
                    print("Task data not found in observation.")
            else:
                print("Observation missing in this sample.")

        print("\nVisualization complete.")
    
    except Exception as e:
        print(f"Error reading .pkl file: {e}")

# Specify the path to your .pkl file
pkl_file_path = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\expert_data\session_20241130_160728\expert_data.pkl"  # Update this with your file path
load_and_visualize_data(pkl_file_path, num_samples=5)

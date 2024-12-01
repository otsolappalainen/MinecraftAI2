import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_and_visualize_data(pkl_file, num_samples=5):
    """Load and visualize data from a .pkl file, focusing on samples with broken blocks."""
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            print(f"Expected a list of samples, got {type(data)}.")
            return
        
        print(f"Loaded {len(data)} total samples from {pkl_file}.")
        
        # Find samples with broken blocks
        samples_with_blocks = []
        for sample in data:
            observation = sample.get('observation', {})
            if observation and 'other' in observation:
                other = observation['other']
                # Check if there are any non-zero values in broken blocks section
                # First 8 values are metadata, rest are broken blocks data
                if len(other) > 8 and np.any(other[8:]):
                    samples_with_blocks.append(sample)
        
        print(f"\nFound {len(samples_with_blocks)} samples with broken blocks.")
        
        if not samples_with_blocks:
            print("No samples with broken blocks found.")
            return
            
        # Display samples with broken blocks
        for idx, sample in enumerate(samples_with_blocks[:num_samples]):
            print(f"\n--- Sample with Broken Blocks {idx + 1} ---")
            action = sample.get('action', None)
            observation = sample.get('observation', {})
            
            print(f"Action: {action}")
            
            if 'image' in observation:
                img = observation['image']
                if isinstance(img, np.ndarray) and img.shape[0] == 3:
                    plt.figure(figsize=(4, 4))
                    plt.imshow(np.transpose(img, (1, 2, 0)))
                    plt.title(f"Sample {idx + 1} - Action: {action}")
                    plt.axis('off')
                    plt.show()
            
            if 'other' in observation:
                other = observation['other']
                if len(other) >= 8:
                    # Display metadata
                    x, z, y = other[0:3]
                    sin_yaw, cos_yaw = other[3:5]
                    health, hunger, alive = other[5:8]
                    
                    print(f"\nPosition: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                    print(f"Health: {health:.2f}, Hunger: {hunger:.2f}, Alive: {alive}")
                    
                    # Display broken blocks data
                    broken_blocks = other[8:]
                    num_blocks = len(broken_blocks) // 4  # 4 values per block
                    
                    print("\nBroken Blocks:")
                    for i in range(num_blocks):
                        block_data = broken_blocks[i*4:(i+1)*4]
                        if np.any(block_data):  # Only show non-zero blocks
                            print(f"Block {i+1}:")
                            print(f"  Type: {block_data[0]:.2f}")
                            print(f"  Position: x={block_data[1]:.2f}, y={block_data[2]:.2f}, z={block_data[3]:.2f}")

    except Exception as e:
        print(f"Error reading .pkl file: {e}")

# Specify the path to your .pkl file
pkl_file_path = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\expert_data\session_20241201_022223\expert_data.pkl"
load_and_visualize_data(pkl_file_path, num_samples=5)

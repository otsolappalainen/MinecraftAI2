import os
import torch as th
import numpy as np
from zipfile import ZipFile
import io

def load_state_dict(path):
    """Load state dict from either .pth or .zip file"""
    if path.endswith('.pth'):
        return th.load(path, map_location='cpu', weights_only=True)
    elif path.endswith('.zip'):
        with ZipFile(path, 'r') as zip_ref:
            # For DQN models, load policy.pth
            with zip_ref.open('policy.pth') as f:
                buffer = io.BytesIO(f.read())
                dqn_dict = th.load(buffer, map_location='cpu', weights_only=True)
                # Extract just the feature extractor part
                fe_dict = {k.replace('features_extractor.', ''): v 
                          for k, v in dqn_dict.items() 
                          if k.startswith('features_extractor.')}
                return fe_dict

def compare_models(bc_path, dqn_path):
    """Compare model weights without loading full models"""
    print(f"\nComparing models:\nBC: {os.path.basename(bc_path)}\nDQN: {os.path.basename(dqn_path)}")
    
    # Load state dicts
    bc_dict = load_state_dict(bc_path)
    dqn_dict = load_state_dict(dqn_path)
    
    # Only compare feature extractor layers
    fe_layers = [
        'scalar_net.0.weight', 'scalar_net.0.bias',
        'scalar_net.3.weight', 'scalar_net.3.bias',
        'image_net.0.weight', 'image_net.0.bias',
        'image_net.1.weight', 'image_net.1.bias',
        'image_net.3.weight', 'image_net.3.bias',
        'image_net.4.weight', 'image_net.4.bias',
        'image_net.6.weight', 'image_net.6.bias',
        'image_net.7.weight', 'image_net.7.bias',
        'fusion_layers.0.weight', 'fusion_layers.0.bias',
        'fusion_layers.3.weight', 'fusion_layers.3.bias'
    ]
    
    comparisons = []
    
    # Compare each layer
    for key in fe_layers:
        if key in bc_dict and key in dqn_dict:
            bc_tensor = bc_dict[key]
            dqn_tensor = dqn_dict[key]
            
            if bc_tensor.shape == dqn_tensor.shape:
                diff = (bc_tensor - dqn_tensor).abs()
                comparisons.append({
                    'layer': key,
                    'mean_diff': diff.mean().item(),
                    'max_diff': diff.max().item(),
                    'std_diff': diff.std().item(),
                    'shape': bc_tensor.shape
                })
            else:
                print(f"Shape mismatch for {key}: BC {bc_tensor.shape} vs DQN {dqn_tensor.shape}")
    
    # Print results
    print("\nLayer-wise differences:")
    print("Layer".ljust(40), "Mean Diff".ljust(12), "Max Diff".ljust(12), "Std Dev".ljust(12), "Shape")
    print("-" * 90)
    
    for comp in comparisons:
        print(
            f"{comp['layer'][:40].ljust(40)}",
            f"{comp['mean_diff']:.6f}".ljust(12),
            f"{comp['max_diff']:.6f}".ljust(12),
            f"{comp['std_diff']:.6f}".ljust(12),
            f"{comp['shape']}"
        )
    
    # Overall statistics
    mean_diffs = [c['mean_diff'] for c in comparisons]
    print("\nOverall Statistics:")
    print(f"Average difference across all layers: {np.mean(mean_diffs):.6f}")
    print(f"Standard deviation of differences: {np.std(mean_diffs):.6f}")
    print(f"Max difference across all layers: {max(c['max_diff'] for c in comparisons):.6f}")

def inspect_model_file(path):
    """Print metadata about model file contents"""
    print(f"\nInspecting: {os.path.basename(path)}")
    
    if path.endswith('.pth'):
        # Load and print PTH contents
        state_dict = th.load(path, map_location='cpu', weights_only=True)
        print("\nPTH file contents:")
        for key in state_dict.keys():
            if isinstance(state_dict[key], th.Tensor):
                print(f"  {key}: shape {state_dict[key].shape}")
            else:
                print(f"  {key}: {type(state_dict[key])}")
                
    elif path.endswith('.zip'):
        # Print ZIP contents
        print("\nZIP file contents:")
        with ZipFile(path, 'r') as zip_ref:
            for info in zip_ref.filelist:
                print(f"  {info.filename}: {info.file_size} bytes")
            
            # Try to load each .pth file and print its contents
            for info in zip_ref.filelist:
                if info.filename.endswith('.pth'):
                    print(f"\nContents of {info.filename}:")
                    with zip_ref.open(info.filename) as f:
                        state_dict = th.load(io.BytesIO(f.read()), map_location='cpu', weights_only=True)
                        for key in state_dict.keys():
                            if isinstance(state_dict[key], th.Tensor):
                                print(f"  {key}: shape {state_dict[key].shape}")
                            else:
                                print(f"  {key}: {type(state_dict[key])}")

def main():
    BC_DIR = "models_bc"
    DQN_DIR = "models_dqn_from_bc"
    
    # List available models
    bc_models = [f for f in os.listdir(BC_DIR) if f.endswith('.pth')]
    dqn_models = [f for f in os.listdir(DQN_DIR) if f.endswith('.zip')]
    
    print("\nAvailable BC models:")
    for idx, name in enumerate(bc_models, 1):
        print(f"{idx}. {name}")
    bc_idx = int(input("Select BC model number: ")) - 1
    bc_path = os.path.join(BC_DIR, bc_models[bc_idx])
    
    print("\nAvailable DQN models:")
    for idx, name in enumerate(dqn_models, 1):
        print(f"{idx}. {name}")
    dqn_idx = int(input("Select DQN model number: ")) - 1
    dqn_path = os.path.join(DQN_DIR, dqn_models[dqn_idx])

    print("\nInspecting model files...")
    inspect_model_file(bc_path)
    inspect_model_file(dqn_path)
    
    # Compare the models
    compare_models(bc_path, dqn_path)

    

if __name__ == "__main__":
    main()
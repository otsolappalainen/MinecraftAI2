import torch
from stable_baselines3 import DQN

# Paths to trained and default models
trained_model_path = r"E:\CNN\model_step_250000.zip"
default_model_path = r"E:\CNN\best_model_20241119-031330.zip"

# Load the trained model
trained_model = DQN.load(trained_model_path)
trained_weights = trained_model.policy.state_dict()

# Load the default model
default_model = DQN.load(default_model_path)  # Assuming you saved a fresh/untrained model
default_weights = default_model.policy.state_dict()

# Compare specific weights (e.g., the 'other' extractor in the policy network)
for key in trained_weights.keys():
    if "features_extractor.other" in key:  # Adjust based on your architecture
        print(f"Layer: {key}")
        print("Trained weights:", trained_weights[key])
        print("Default weights:", default_weights[key])
        print("Weight difference:", (trained_weights[key] - default_weights[key]).norm().item())

# Visualize how the model processes specific inputs
import numpy as np

# Generate a sample input
sample_input = {
    "image": torch.tensor(np.random.rand(1, 1, 224, 224), dtype=torch.float32),
    "other": torch.tensor(np.random.rand(1, 28), dtype=torch.float32)
}

# Get the feature extractor from the trained model
feature_extractor = trained_model.policy.features_extractor

# Pass the sample input through the feature extractor
output = feature_extractor(sample_input)
print("Feature extractor output:", output)
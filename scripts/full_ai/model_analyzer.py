import os
import torch
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym


def generate_random_observations(obs_space, n_samples):
    """
    Generate random observations that match the expected format of the observation space.
    """
    obs_samples = []
    for _ in range(n_samples):
        sample = {}
        for key, space in obs_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                # Ensure proper shape and dtype for Box spaces
                sample[key] = np.random.uniform(
                    low=space.low,
                    high=space.high,
                    size=space.shape
                ).astype(space.dtype)
            elif isinstance(space, gym.spaces.Discrete):
                # Discrete spaces as integers
                sample[key] = np.array([space.sample()], dtype=np.int32)
            elif isinstance(space, gym.spaces.MultiDiscrete):
                # MultiDiscrete spaces
                sample[key] = space.sample()
            else:
                raise ValueError(f"Unsupported space type: {type(space)}")
        obs_samples.append(sample)
    return obs_samples


def analyze_saved_model(model_path):
    try:
        # Load the model
        model = PPO.load(model_path)

        # Access the policy network
        policy = model.policy

        print(f"\n--- Analyzing Model: {os.path.basename(model_path)} ---")

        # Print model parameters
        print("\nPolicy Network Parameters:")
        #for name, param in policy.named_parameters():
            #print(f"{name}: {param.shape}")

        # Analyze the action distribution
        # Since we need observations to get the action distribution, we'll generate some random observations
        obs_space = model.observation_space

        # Generate a batch of random observations
        n_samples = 100
        obs_samples = generate_random_observations(obs_space, n_samples)

        # Convert observations to tensors
        obs_tensor, _ = policy.obs_to_tensor({k: np.stack([obs[k] for obs in obs_samples]) for k in obs_samples[0]})

        # Get action distributions
        with torch.no_grad():
            distribution = policy.get_distribution(obs_tensor)
            actions = distribution.get_actions()
            log_probs = distribution.log_prob(actions)
            entropies = distribution.entropy()

        # Convert actions to numpy
        actions = actions.cpu().numpy()

        # Analyze action probabilities
        unique_actions, counts = np.unique(actions, return_counts=True)
        action_probabilities = counts / n_samples

        #print("\nAction Probabilities from Random Observations:")
        #for action, prob in zip(unique_actions, action_probabilities):
            #print(f"Action {action}: Probability {prob}")

        # Analyze the entropy
        avg_entropy = entropies.mean().item()
        print(f"\nAverage Policy Entropy: {avg_entropy}")

        # Analyze value function predictions
        values = policy.predict_values(obs_tensor)
        avg_value = values.mean().item()
        print(f"\nAverage Value Function Prediction: {avg_value}")

    except Exception as e:
        print(f"Error analyzing model {model_path}: {e}")


def analyze_all_models(models_dir):
    # List all .zip files in the directory
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".zip")]

    if not model_files:
        print("No model files found in the specified directory.")
        return

    # Analyze each model
    for model_path in model_files:
        analyze_saved_model(model_path)


if __name__ == '__main__':
    models_dir = r'C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\models'  # Replace with your models' folder
    analyze_all_models(models_dir)
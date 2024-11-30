# File: evaluate_bc_model.py

import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time

# Import the MinecraftEnv environment
from mod_env_v1 import MinecraftEnv

# Define the model architecture (same as in your training code)
class FullModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FullModel, self).__init__()
        # Scalar observation processing
        scalar_input_dim = observation_space['other']
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        scalar_output_size = 128

        # Image processing
        image_input_channels = observation_space['image'][0]  # Should be 3 for RGB images
        self.image_net = nn.Sequential(
            nn.Conv2d(image_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size
        with th.no_grad():
            dummy_input = th.zeros(1, image_input_channels, 224, 224)
            conv_output_size = self.image_net(dummy_input).shape[1]

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(scalar_output_size + conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._features_dim = 128  # Final feature dimension

        # Output layer
        self.action_head = nn.Linear(self._features_dim, action_space)

    def forward(self, observations):
        # Concatenate 'other' and 'task'
        other_combined = th.cat([observations['other'], observations['task']], dim=1)
        scalar_features = self.scalar_net(other_combined)
        image_features = self.image_net(observations['image'])
        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        action_logits = self.action_head(fused_features)
        return action_logits

def main():
    # Device configuration
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Load the trained model
    MODEL_PATH = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\models_bc\final_model.pt"

    # Define observation and action spaces based on your data
    observation_space = {
        'image': (3, 224, 224),
        'other': 8 + 20  # 'other' length (8) plus 'task' length (20)
    }
    action_space = 18  # Number of actions

    # Initialize the model
    model = FullModel(observation_space, action_space).to(device)
    model.load_state_dict(th.load(MODEL_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Define the task (same as in the environment)
    task = np.array([0, 1] + [0]*18, dtype=np.float32)  # Example task

    # Initialize the environment with the task
    env = MinecraftEnv(task=task)

    # Reset the environment
    obs, info = env.reset()

    # Start the evaluation loop
    num_episodes = 1
    max_steps_per_episode = 1000

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps_per_episode:
            # Prepare the observation
            image = obs['image']  # Shape (3, 224, 224)
            other = obs['other']  # Shape (8,)
            task_obs = obs['task']  # Shape (20,)
            # No need to concatenate here since the model does it

            # Convert to tensors and add batch dimension
            image_tensor = th.tensor(image, dtype=th.float32).unsqueeze(0).to(device)  # Shape (1, 3, 224, 224)
            other_tensor = th.tensor(other, dtype=th.float32).unsqueeze(0).to(device)  # Shape (1, 8)
            task_tensor = th.tensor(task_obs, dtype=th.float32).unsqueeze(0).to(device)  # Shape (1, 20)

            observations = {'image': image_tensor, 'other': other_tensor, 'task': task_tensor}

            # Get action logits from the model
            with th.no_grad():
                action_logits = model(observations)
                action_probs = th.softmax(action_logits, dim=1)
                action = th.argmax(action_probs, dim=1).item()

            # Take the action in the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Optional: Print some information
            print(f"Step {steps}: Action {action}, Reward {reward}, Total Reward {total_reward}")

            # Sleep to limit the speed
            time.sleep(0.05)

        print(f"Episode {episode+1} finished after {steps} steps with total reward {total_reward}")

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()

# File: evaluate_bc_model.py

import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time
import asyncio

# Import the MinecraftEnv environment
from mod_env_v1 import MinecraftEnv

# Add these functions at the top of evaluate_bc_model.py

def validate_observation(obs, episode, step):
    """Validate observation values and shapes"""
    try:
        # Check for NaN/Inf values
        for key, value in obs.items():
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                print(f"Warning: {key} contains NaN/Inf values at episode {episode}, step {step}")
        
        # Value ranges
        if not (0 <= obs['image'].min() <= obs['image'].max() <= 255):
            print(f"Warning: Image values outside expected range at episode {episode}, step {step}")
            
        print(f"Observation ranges - Image: [{obs['image'].min():.2f}, {obs['image'].max():.2f}], "
              f"Other: [{obs['other'].min():.2f}, {obs['other'].max():.2f}]")
              
    except Exception as e:
        print(f"Validation error: {str(e)}")

def print_action_probs(probs, action):
    """Print action probabilities distribution"""
    probs = probs.cpu().numpy()[0]
    sorted_actions = np.argsort(-probs)  # Sort in descending order
    print("\nTop 5 actions and their probabilities:")
    for i in range(5):
        a = sorted_actions[i]
        p = probs[a]
        star = "*" if a == action else " "
        print(f"{star}Action {a}: {p:.4f}")

def analyze_dataset(dataset):
    """Analyze action distribution in training data"""
    action_counts = {}
    total = 0
    
    for i in range(len(dataset)):
        _, action = dataset[i]
        action = action.item()
        action_counts[action] = action_counts.get(action, 0) + 1
        total += 1
    
    print("\nAction distribution in training data:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / total) * 100
        print(f"Action {action}: {count} ({percentage:.2f}%)")

def normalize_observations(obs):
    """Normalize observation values to reasonable ranges"""
    normalized = {}
    
    # Normalize image (already 0-1)
    normalized['image'] = obs['image']
    
    # Create copy of other observations
    normalized['other'] = obs['other'].copy()
    
    # Normalize yaw to [-180, 180]
    yaw_idx = 0  # Assuming yaw is first value in 'other'
    normalized['other'][yaw_idx] = ((normalized['other'][yaw_idx] + 180) % 360) - 180
    
    # Normalize yaw to [-1, 1]
    normalized['other'][yaw_idx] = normalized['other'][yaw_idx] / 180.0
    
    return normalized

class ObservationStats:
    def __init__(self):
        self.step_stats = []
        
    def update(self, obs, action_probs, action):
        stats = {
            'yaw': obs['other'][0],
            'action': action,
            'action_entropy': -(action_probs * action_probs.log()).sum().item(),
            'max_prob': action_probs.max().item()
        }
        self.step_stats.append(stats)
        
    def print_summary(self):
        print("\nObservation Statistics:")
        print(f"Yaw range: [{min(s['yaw'] for s in self.step_stats):.2f}, {max(s['yaw'] for s in self.step_stats):.2f}]")
        print(f"Action entropy range: [{min(s['action_entropy'] for s in self.step_stats):.4f}, {max(s['action_entropy'] for s in self.step_stats):.4f}]")
        
        action_counts = {}
        for s in self.step_stats:
            action_counts[s['action']] = action_counts.get(s['action'], 0) + 1
        
        print("\nAction distribution during evaluation:")
        for action, count in sorted(action_counts.items()):
            print(f"Action {action}: {count} times ({count/len(self.step_stats)*100:.1f}%)")

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

        if observations['image'].ndimension() == 4:  # Check if the input is in the wrong shape [batch_size, height, width, channels]
            observations['image'] = observations['image'].permute(0, 3, 1, 2)  # Change to [batch_size, channels, height, width]

        other_combined = th.cat([observations['other'], observations['task']], dim=1)
        scalar_features = self.scalar_net(other_combined)
        image_features = self.image_net(observations['image'])


        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        action_logits = self.action_head(fused_features)
        return action_logits

async def main():
    # Device configuration
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Load the trained model
    MODEL_PATH = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\models_bc\model_epoch_10.pth"

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

    # Reset the environment (this should be awaited)
    obs, info = await env.reset()  # Await the reset method

    # Start the evaluation loop
    episodes = 5  # Number of episodes to run
    max_steps_per_episode = 50
    
    stats = ObservationStats()  # Initialize observation statistics

    for episode in range(episodes):
        obs, info = await env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Normalize observations
            norm_obs = normalize_observations(obs)
            
            # Validate observations
            validate_observation(norm_obs, episode, steps)
            
            # Prepare input for model
            observations = {
                'image': th.FloatTensor(norm_obs['image']).unsqueeze(0).to(device),
                'other': th.FloatTensor(norm_obs['other']).unsqueeze(0).to(device),
                'task': th.FloatTensor(task).unsqueeze(0).to(device)
            }
            
            # Get model prediction
            with th.no_grad():
                action_logits = model(observations)
                temperature = 1.0  # Define temperature value
                action_probs = th.softmax(action_logits / temperature, dim=1)  # temperature > 1 makes distribution more uniform
                action = th.argmax(action_probs, dim=1).item()
                
                # Track statistics
                stats.update(norm_obs, action_probs[0], action)
                
                # Print action probabilities
                print_action_probs(action_probs, action)
            
            # Take action
            obs, reward, done, truncated, info = await env.step(action)
            total_reward += reward
            steps += 1

            print(f"Step {steps}: Action {action}, Reward {reward}, Total Reward {total_reward}")
            time.sleep(0.2)

        print(f"Episode {episode+1} finished after {steps} with total reward {total_reward}")

    stats.print_summary()  # Print observation statistics summary

    await env.close()

if __name__ == '__main__':
    asyncio.run(main())

import os
import numpy as np
from stable_baselines3 import PPO
from minecraft_env import MinecraftEnv
from colorama import Fore, Style

# Load the model and environment
model_path = "ppo_minecraft_agent.zip"
env = MinecraftEnv()

# Check if model file exists and load
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, env=env)
else:
    print("Model file not found.")
    exit()

# Print model configuration and hyperparameters
print("\nModel Configuration and Hyperparameters:")
print(model.policy)

# Print last layer weights and biases for policy and value networks
print("\nLast Layer Weights and Biases:")
print("Policy Network Last Layer Weights:")
print(model.policy.mlp_extractor.policy_net[-2].weight)
print("Policy Network Last Layer Biases:")
print(model.policy.mlp_extractor.policy_net[-2].bias)

print("Value Network Last Layer Weights:")
print(model.policy.mlp_extractor.value_net[-2].weight)
print("Value Network Last Layer Biases:")
print(model.policy.mlp_extractor.value_net[-2].bias)

# Evaluate model for a few episodes
num_eval_episodes = 5
episode_rewards = []
print("\nEvaluating the Model:")

for episode in range(num_eval_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        # Check if state coordinates are available
        if env.agent.state is not None:
            x, z = env.agent.state
            yaw, pitch = env.agent.yaw, env.agent.pitch
            print(
                f"Action: {action} | "
                f"Coordinates: {Fore.GREEN}X={x:.2f}{Style.RESET_ALL}, "
                f"{Fore.BLUE}Z={z:.2f}{Style.RESET_ALL}, "
                f"{Fore.CYAN}Yaw={yaw:.2f}{Style.RESET_ALL}, "
                f"{Fore.MAGENTA}Pitch={pitch:.2f}{Style.RESET_ALL} | "
                f"Reward: {Fore.YELLOW}{reward}{Style.RESET_ALL} | "
                f"Done: {Fore.RED}{done}{Style.RESET_ALL}"
            )
        else:
            print("State not available. Skipping coordinates output.")

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1} ended with total reward: {total_reward}")

# Calculate average reward
average_reward = np.mean(episode_rewards)
print(f"\nAverage Reward over {num_eval_episodes} episodes: {average_reward}")

# Close the environment
env.close()


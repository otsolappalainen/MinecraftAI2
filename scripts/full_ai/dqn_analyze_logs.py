import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import torch

from env_dqn import MinecraftEnv  # Import the updated environment

MODELS_DIR = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\models_dqn"

def list_models():
    """
    List all models in the models directory.
    """
    if not os.path.exists(MODELS_DIR):
        print(f"Models directory '{MODELS_DIR}' does not exist!")
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]

def load_model(model_name):
    """
    Load the RL model from the given path.
    Supports DQN and PPO for now.
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model path '{model_path}' does not exist!")

    # Automatically determine model type based on file name
    if "dqn" in model_name.lower():
        model_class = DQN
    elif "ppo" in model_name.lower():
        model_class = PPO
    else:
        raise ValueError("Model type not recognized. Ensure 'dqn' or 'ppo' is in the filename.")

    print(f"Loading model: {model_path}")
    return model_class.load(model_path)

def plot_action_distribution(env, model, num_episodes=100):
    """
    Visualize the distribution of actions taken by the agent.
    """
    print("Collecting action data...")
    action_counts = np.zeros(env.action_space.n)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[action] += 1
            obs, _, done, truncated, _ = env.step(action)
            done = done or truncated

    plt.figure(figsize=(10, 6))
    plt.bar(range(env.action_space.n), action_counts, tick_label=[str(i) for i in range(env.action_space.n)])
    plt.title(f"Action Distribution Over {num_episodes} Episodes")
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.show()

def visualize_policy_outputs(env, model, num_samples=100):
    """
    Sample random observations and visualize the policy's outputs (Q-values for DQN, probabilities for PPO).
    """
    print("Sampling random observations...")
    sampled_obs = []
    for _ in range(num_samples):
        obs, _ = env.reset()
        sampled_obs.append(obs)

    policy_outputs = []
    for obs in sampled_obs:
        if isinstance(model, DQN):
            q_values = model.policy.q_net(torch.tensor([obs], dtype=torch.float32).to(model.device)).detach().cpu().numpy()
            policy_outputs.append(q_values)
        elif isinstance(model, PPO):
            policy_output = model.policy.forward(torch.tensor([obs], dtype=torch.float32).to(model.device))
            policy_outputs.append(policy_output.logits.detach().cpu().numpy())

    policy_outputs = np.array(policy_outputs).squeeze()

    plt.figure(figsize=(10, 6))
    for i in range(env.action_space.n):
        plt.hist(policy_outputs[:, i], bins=30, alpha=0.7, label=f"Action {i}")
    plt.title(f"Policy Output Distribution ({num_samples} samples)")
    plt.xlabel("Output Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def evaluate_model(env, model, num_episodes=10):
    """
    Evaluate the model on a given number of episodes and display results.
    """
    print(f"Evaluating model over {num_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

def main():
    # List available models
    models = list_models()
    if not models:
        print("No models found in the directory. Exiting.")
        return

    print("\nAvailable Models:")
    for idx, model_name in enumerate(models):
        print(f"{idx + 1}: {model_name}")

    # Prompt user to select a model
    selected_idx = int(input(f"Select a model to analyze (1-{len(models)}): ")) - 1
    if selected_idx < 0 or selected_idx >= len(models):
        print("Invalid selection. Exiting.")
        return

    model_name = models[selected_idx]
    model = load_model(model_name)

    # Initialize the environment
    env = MinecraftEnv()

    # Perform analysis
    while True:
        print("\nOptions:")
        print("1. Evaluate model performance")
        print("2. Plot action distribution")
        print("3. Visualize policy outputs")
        print("4. Exit")
        choice = input("Select an option (1-4): ")

        if choice == "1":
            num_episodes = int(input("Number of evaluation episodes: "))
            evaluate_model(env, model, num_episodes)
        elif choice == "2":
            num_episodes = int(input("Number of episodes for action distribution: "))
            plot_action_distribution(env, model, num_episodes)
        elif choice == "3":
            num_samples = int(input("Number of random samples for policy visualization: "))
            visualize_policy_outputs(env, model, num_samples)
        elif choice == "4":
            print("Exiting analysis tool.")
            break
        else:
            print("Invalid choice. Please select an option between 1 and 4.")

if __name__ == "__main__":
    main()
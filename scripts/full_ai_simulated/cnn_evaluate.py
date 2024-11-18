import os
import logging
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from cnn_env import SimulatedEnvGraphics  # Import the environment

# Hardcoded paths and parameters
MODEL_PATH = r"E:\CNN"  # Path to the main directory
BEST_MODELS_PATH = MODEL_PATH  # Path to best models
PARALLEL_ENVS = 1  # Number of parallel environments
RENDER_MODE = "none"  # Enable rendering for evaluation
EVALUATION_EPISODES = 100  # Number of episodes to evaluate

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def make_env(render_mode="none"):
    """
    Create a simulated environment instance.
    """
    def _init():
        return SimulatedEnvGraphics(render_mode=render_mode)
    return _init


def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate a pre-trained model on the environment.
    """
    logger.info(f"Evaluating model for {num_episodes} episodes...")
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = [False] * PARALLEL_ENVS
        episode_reward = 0

        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            episode_reward += rewards[0]
            done = dones

        total_rewards.append(episode_reward)
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward}")

    mean_reward = np.mean(total_rewards)
    logger.info(f"Mean Reward over {num_episodes} episodes: {mean_reward}")
    return mean_reward


def main():
    print("Starting evaluation...")
    print(f"Using device: {'cuda' if th.cuda.is_available() else 'cpu'}")

    # Ensure the best models directory exists
    if not os.path.exists(BEST_MODELS_PATH):
        print(f"Directory does not exist: {BEST_MODELS_PATH}")
        return

    # Check for existing models
    existing_models = [
        file for file in os.listdir(BEST_MODELS_PATH) if file.endswith(".zip")
    ]

    if not existing_models:
        print(f"No models found in {BEST_MODELS_PATH}. Please ensure the directory contains valid `.zip` files.")
        return

    print("Available models:")
    for idx, model_name in enumerate(existing_models):
        print(f"{idx + 1}. {model_name}")

    # User selection
    while True:
        try:
            choice = int(input("Enter the number of the model to evaluate: "))
            if 1 <= choice <= len(existing_models):
                break
            else:
                print(f"Please select a number between 1 and {len(existing_models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    selected_model = os.path.join(BEST_MODELS_PATH, existing_models[choice - 1])
    print(f"Loading model: {selected_model}")

    # Create the environment
    try:
        env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])
    except Exception as e:
        logger.error(f"Error initializing the environment: {e}")
        return

    # Load the model
    try:
        model = DQN.load(selected_model, env=env, device="cuda" if th.cuda.is_available() else "cpu")
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        return

    print("Model loaded successfully. Starting evaluation...")

    # Validate observation and action space
    print(f"Environment observation space: {env.observation_space}")
    print(f"Model observation space: {model.policy.observation_space}")
    print(f"Environment action space: {env.action_space}")
    print(f"Model action space: {model.policy.action_space}")

    # Evaluate the model
    try:
        evaluate_model(model, env, num_episodes=EVALUATION_EPISODES)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

    # Close the environment
    env.close()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An unexpected error occurred:", e)
        import traceback
        traceback.print_exc()
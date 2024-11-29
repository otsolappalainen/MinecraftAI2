# File: evaluate_mod_env.py

import os
import numpy as np
import torch as th
from stable_baselines3 import DQN
import logging
import gymnasium as gym

# Suppress TensorFlow logs if TensorFlow is not needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO logs

# Import your environment
from mod_env_v1 import MinecraftEnv  # Use the updated MinecraftEnv

# Configuration Constants
MODEL_DIR = "gen5_models"  # Directory containing trained models
EVAL_MODEL_NAME = "best_model.zip"  # Name of the model file to evaluate
LOG_FILE = "evaluation_data.csv"  # Log file to store evaluation data
NUM_EVAL_EPISODES = 10  # Number of episodes to run during evaluation
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode to prevent infinite loops

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize Logging
def init_logging(log_file, debug=True):
    logger = logging.getLogger('EvaluationLogger')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file, mode='w')  # Overwrite existing log file
    c_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    f_handler.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

# Create the logger
logger = init_logging(LOG_FILE, debug=True)

def load_model(model_dir, model_name, device):
    """
    Load the trained model from the specified directory.

    Args:
        model_dir (str): Directory where models are stored.
        model_name (str): Filename of the model to load.
        device (torch.device): Device to load the model on.

    Returns:
        DQN: Loaded DQN model.
    """
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' does not exist.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    logger.info(f"Loading model from '{model_path}'...")
    model = DQN.load(model_path, device=device)
    logger.info("Model loaded successfully.")
    return model

def evaluate(model, env, num_episodes, max_steps):
    """
    Evaluate the trained model on the environment.

    Args:
        model (DQN): Trained DQN model.
        env (gym.Env): Gymnasium environment instance.
        num_episodes (int): Number of episodes to run.
        max_steps (int): Maximum steps per episode.
    """
    for episode in range(1, num_episodes + 1):
        # Reset the environment and get initial observation
        observation, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        logger.info(f"Starting Episode {episode}/{num_episodes}")

        while not (terminated or truncated):
            step_count += 1
            if step_count > max_steps:
                logger.warning(f"Episode {episode} reached maximum steps ({max_steps}). Truncating.")
                truncated = True
                break

            try:
                # Get action from the model
                action, _states = model.predict(observation, deterministic=True)

                # Log the chosen action
                action_mapping = env._get_action_mapping()
                action_index = int(action)  # action is scalar
                action_name = action_mapping.get(action_index, "no_op")
                logger.debug(f"Action chosen: {action_index} ('{action_name}')")

                # Take a step in the environment
                observation, reward, terminated, truncated, info = env.step(action_index)

                # Accumulate reward
                total_reward += reward

            except Exception as e:
                logger.error(f"An error occurred during evaluation: {e}")
                break

        logger.info(f"Episode {episode} ended. Total Reward: {total_reward:.2f}, Steps: {step_count}")
        logger.info("-" * 50)

def main():
    # Device Configuration
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        th.cuda.set_device(0)  # Ensure GPU 0 is used
        th.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking for speed
        logger.info("Using CUDA for evaluation.")
    else:
        logger.info("Using CPU for evaluation.")

    # Load the trained model
    try:
        model = load_model(MODEL_DIR, EVAL_MODEL_NAME, device)
    except FileNotFoundError as fe:
        logger.error(f"Evaluation aborted: {fe}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}")
        return

    # Initialize the environment
    try:
        env = MinecraftEnv()
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return

    # Run evaluation without wrapping with DummyVecEnv
    try:
        evaluate(model, env, NUM_EVAL_EPISODES, MAX_STEPS_PER_EPISODE)
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
    finally:
        # Close the environment
        env.close()
        logger.info("Evaluation completed and environment closed.")

if __name__ == "__main__":
    main()



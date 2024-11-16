import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import argparse
from pynput import keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.monitor import Monitor
from env import MinecraftEnv  # Import your environment

# Set up logging
logger = logging.getLogger(__name__)

# Flags to control evaluation
start_evaluation = False

def on_press_start(key):
    """
    Start evaluation when the user presses a specific key.
    """
    global start_evaluation
    try:
        logger.debug(f"Key pressed: {key}")
        if hasattr(key, 'char') and key.char == 'e':  # 'e' key to start evaluation
            logger.info("Starting evaluation: 'E' key pressed.")
            start_evaluation = True
            return False  # Stop listener
    except AttributeError as e:
        logger.error(f"Error in on_press_start: {e}")


def wait_for_start_key():
    """
    Wait for the user to press the 'e' key to start the evaluation.
    """
    logger.info("Waiting for the 'e' key to start evaluation...")
    try:
        with keyboard.Listener(on_press=on_press_start) as listener:
            listener.join()
    except Exception as e:
        logger.error(f"Exception in wait_for_start_key: {e}")
    logger.info("'E' key pressed. Proceeding with evaluation...")


def choose_model_to_load(env, models_dir):
    """
    Allow the user to select a model to load for evaluation.
    """
    models = [f for f in os.listdir(models_dir) if f.endswith(".zip")]

    if not models:
        logger.info(f"No pre-trained models found in {models_dir}. Exiting.")
        return None

    # Sort models to ensure consistent ordering
    models.sort()
    # Default to the "base default" model (assuming the first one)
    default_model = models[0]

    print("Available models:")
    for idx, model_name in enumerate(models):
        print(f"{idx}: {model_name}")
    
    choice = input(f"Enter the number of the model you want to load or press Enter to load the default model '{default_model}': ")
    if choice == '':
        model_path = os.path.join(models_dir, default_model)
    else:
        try:
            model_idx = int(choice)
            if model_idx < 0 or model_idx >= len(models):
                print("Invalid selection. Exiting evaluation.")
                return None
            model_path = os.path.join(models_dir, models[model_idx])
        except ValueError:
            print("Invalid input. Exiting evaluation.")
            return None

    logger.info(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Check if the model's observation space matches the environment's
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        model.set_env(env)
        logger.info("Loaded model is compatible with the observation space.")
    except ValueError:
        logger.warning("Loaded model has incompatible observation space. Exiting.")
        return None

    return model


def evaluate_model(model, env, num_episodes=10):
    """
    Evaluate the given model on the environment.
    """
    logger.info("Starting evaluation...")
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            # Predict the action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Log each step
            logger.debug(f"Episode {episode + 1}, Step {step_count}, Reward: {reward:.2f}, Cumulative Reward: {episode_reward:.2f}")

            # Break if the episode is truncated
            if truncated:
                break

        episode_rewards.append(episode_reward)
        logger.info(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}")

    avg_reward = np.mean(episode_rewards)
    logger.info(f"Evaluation completed over {num_episodes} episodes.")
    logger.info(f"Average Reward: {avg_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging level
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    logger.info("Initializing environment...")
    try:
        env = MinecraftEnv()
        env = Monitor(env, filename="./logs_eval/")
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        sys.exit(1)

    # Directory where models are saved
    models_dir = r'C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\models'

    # Allow user to choose a model to load
    model = choose_model_to_load(env, models_dir)

    if model is None:
        logger.error("No model loaded. Exiting.")
        sys.exit(1)

    # Wait for the user to press the 'e' key to start evaluation
    wait_for_start_key()

    # Evaluate the model
    num_episodes = 10  # Set the number of evaluation episodes
    evaluate_model(model, env, num_episodes=num_episodes)

    logger.info("Closing environment...")
    env.close()

    logger.info("Evaluation process has been completed.")


if __name__ == "__main__":
    main()
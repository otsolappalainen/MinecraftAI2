# evaluate_agent_dqn.py

import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import threading
import argparse
from pynput import keyboard
import traceback
import logging
import time
from datetime import datetime
import re

from stable_baselines3 import DQN
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.monitor import Monitor

from env_dqn import MinecraftEnv  # Import the updated environment

# -------------------- Configuration --------------------

# Directories
MODELS_DIR = r'C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\test_models'
REWARD_MODEL = "constant_x"  # or "random_task"

# -------------------- Logging Setup --------------------

# Create custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set your own code's logging level

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # Console shows INFO level

file_handler = logging.FileHandler('evaluation.log')
file_handler.setLevel(logging.DEBUG)  # File logs DEBUG level

# Create formatters and add them to handlers
console_format = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s', datefmt='%H:%M:%S')
file_format = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configure logging for external libraries
logging.getLogger('stable_baselines3').setLevel(logging.DEBUG)  # Detailed logs from SB3
logging.getLogger('gym').setLevel(logging.WARNING)  # Suppress gym warnings

# -------------------- Global Flags --------------------

start_evaluation = False
stop_evaluation = False

# -------------------- Exception Handling --------------------

def handle_thread_exception(args):
    """Log uncaught exceptions in threads."""
    if issubclass(args.exc_type, KeyboardInterrupt):
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
        return
    logger.error("Uncaught exception in thread:", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

threading.excepthook = handle_thread_exception

# -------------------- Custom Feature Extractor --------------------

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # Extract shapes from observation space
        self.image_size = 224 * 224
        self.position_size = 5  # x, y, z, yaw, pitch
        self.task_size = 20  # As defined in the environment
        self.scalar_size = 3  # health, hunger, alive

        # CNN for image input
        n_input_channels = 1  # Grayscale images
        self.image_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=False),
            nn.Flatten(),
        )

        # Compute the flattened size of image features
        with torch.no_grad():
            sample_image = torch.zeros((1, n_input_channels, 224, 224))
            image_features = self.image_cnn(sample_image)
            self.image_feature_dim = image_features.shape[1]

        # Fully connected layers for position and task
        self.position_task_net = nn.Sequential(
            nn.Linear(self.position_size + self.task_size, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
        )

        # Fully connected layers for scalar inputs
        self.scalar_net = nn.Sequential(
            nn.Linear(self.scalar_size, 16),  # Health, hunger, alive
            nn.ReLU(inplace=False),
            nn.Linear(16, 16),
            nn.ReLU(inplace=False),
        )

        # Compute total feature dimension
        self._features_dim = self.image_feature_dim + 32 + 16
        logger.info(f"CustomCombinedExtractor initialized with features_dim={self._features_dim}")

    def forward(self, observations):
        # Ensure observations is a torch tensor
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations).float()

        # Process image input
        image_obs = observations[:, :self.image_size]
        image_obs = image_obs.view(-1, 1, 224, 224)  # Reshape to [batch_size, channels, height, width]
        image_features = self.image_cnn(image_obs)

        # Process position and task
        start = self.image_size
        end = start + self.position_size + self.task_size
        pos_task_obs = observations[:, start:end]
        position_task_features = self.position_task_net(pos_task_obs)

        # Process scalar inputs
        scalar_obs = observations[:, -self.scalar_size:]
        scalar_features = self.scalar_net(scalar_obs)

        # Concatenate all features
        features = torch.cat((image_features, position_task_features, scalar_features), dim=1)
        return features

# -------------------- Keyboard Control --------------------

def on_press_stop(key):
    global stop_evaluation
    try:
        if hasattr(key, 'char') and key.char == 'i':
            logger.info("Stopping evaluation: 'i' key pressed.")
            stop_evaluation = True
            return False  # Stop listener
    except AttributeError as e:
        logger.error(f"Error in on_press_stop: {e}")

def on_press_start(key):
    global start_evaluation
    try:
        if hasattr(key, 'char') and key.char == 'o':
            logger.info("Starting evaluation: 'o' key pressed.")
            start_evaluation = True
            return False  # Stop listener
    except AttributeError as e:
        logger.error(f"Error in on_press_start: {e}")

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press_stop)
    listener.start()
    return listener

def wait_for_start_key():
    logger.info("Waiting for the 'o' key to start evaluation...")
    try:
        with keyboard.Listener(on_press=on_press_start) as listener:
            listener.join()
    except Exception as e:
        logger.error(f"Exception in wait_for_start_key: {e}")
    logger.info("'o' key pressed. Proceeding with evaluation...")

# -------------------- Model Loading --------------------

def choose_model_to_load(env, models_dir):
    models = [f for f in os.listdir(models_dir) if f.endswith(".zip")]

    if not models:
        logger.error(f"No pre-trained models found in {models_dir}. Exiting.")
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
                print("Invalid selection. Exiting.")
                return None
            model_path = os.path.join(models_dir, models[model_idx])
        except ValueError:
            print("Invalid input. Exiting.")
            return None

    logger.info(f"Loading model from {model_path}...")
    model = DQN.load(model_path, env=env)

    # Check if the model's observation space matches the environment's
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        model.set_env(env)
        logger.info("Loaded model is compatible with the new observation space.")
    except ValueError:
        logger.error("Loaded model has incompatible observation space. Exiting.")
        return None
    return model

# -------------------- Custom DQN Policy --------------------

from stable_baselines3.dqn.policies import DQNPolicy

class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs)

# -------------------- Main Evaluation Function --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualization is not implemented in this script")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--scenario", type=str, default="default", help="Select evaluation scenario: default or debug")
    args = parser.parse_args()

    # Update logging level if debug flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        global DEBUG
        DEBUG = True

    logger.info("Initializing environment...")
    try:
        env = MinecraftEnv(reward_model=REWARD_MODEL)
        env = Monitor(env)
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        traceback.print_exc()
        return

    torch.autograd.set_detect_anomaly(True)

    # Allow user to choose model to load
    model = choose_model_to_load(env, MODELS_DIR)

    if model is None:
        logger.error("No model loaded. Exiting.")
        return

    logger.info("Waiting for 'o' key to start evaluation...")
    wait_for_start_key()

    logger.info("Starting keyboard listener for stopping evaluation...")
    listener = start_keyboard_listener()

    obs, _ = env.reset()  # Extract only `obs` from the tuple
    total_reward = 0.0

    logger.info("Starting evaluation...")
    start_time = time.time()

    while not stop_evaluation:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)  # Updated to handle five return values
        total_reward += reward

        if terminated or truncated:  # Check for episode end conditions
            obs, _ = env.reset()  # Reset environment when episode ends

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Total reward during evaluation: {total_reward}")

    logger.info("Closing environment and stopping listener...")
    env.close()
    listener.stop()

    logger.info("Evaluation process has been stopped.")
    sys.exit(0)
if __name__ == "__main__":
    main()

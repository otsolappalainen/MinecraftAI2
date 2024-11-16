# train_agent_dqn.py

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

from stable_baselines3 import DQN
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList



from env_dqn import MinecraftEnv  # Import the updated environment

# -------------------- Configuration --------------------

# Learning rate
LEARNING_RATE = 2e-4  # Reduced learning rate for stability

# Directories
MODELS_DIR = r'C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\models_dqn'
LOGS_DIR = "./logs_dqn/"
LOGS_EVAL_DIR = "./logs_eval_dqn/"
TENSORBOARD_LOG_DIR = "./dqn_minecraft_tensorboard/"
BEST_MODEL_DIR = os.path.join(LOGS_DIR, "best_model")
TIMESTAMPED_BEST_MODELS_DIR = os.path.join(LOGS_DIR, "timestamped_best_models")

# Training parameters
TRAINING_PARAMS = {
    'total_timesteps': 500000,  # Increased total timesteps for better learning
    'learning_rate': LEARNING_RATE,
    'buffer_size': 30000,  # Reduced buffer size to manage memory usage
    'learning_starts': 1000,
    'batch_size': 128,  # Increased batch size for stability
    'gamma': 0.99,
    'train_freq': (1, 'step'),  # Train every 4 steps
    'target_update_interval': 1000,
    'exploration_fraction': 0.3,  # Increased exploration
    'exploration_final_eps': 0.1,
    'verbose': 1
}

# Reward model
REWARD_MODEL = "constant_x"  # or "random_task"

# Other configurations
DEBUG = False  # Set to True for debug logging

# -------------------- Logging Setup --------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

# -------------------- Global Flags --------------------

start_training = False
stop_training = False

# -------------------- Exception Handling --------------------

def handle_thread_exception(args):
    """Log uncaught exceptions in threads."""
    if issubclass(args.exc_type, KeyboardInterrupt):
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
        return
    logger.error("Uncaught exception in thread:", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

threading.excepthook = handle_thread_exception

# -------------------- Custom Callbacks --------------------

class TimestampedCheckpointCallback(BaseCallback):
    """
    A custom callback to save models with a unique timestamp every `save_freq` steps.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(TimestampedCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.save_path, f"dqn_minecraft_{timestamp}.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                logger.info(f"Model checkpoint saved at: {model_path}")
        return True






class TimestampedEvalCallback(EvalCallback):
    def __init__(self, *args, save_path, **kwargs):
        super(TimestampedEvalCallback, self).__init__(*args, **kwargs)
        self.save_path = save_path

    def _on_step(self) -> bool:
        result = super(TimestampedEvalCallback, self)._on_step()
        if self.best_model_save_path is not None and os.path.exists(
            os.path.join(self.best_model_save_path, "best_model.zip")
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(self.save_path, f"best_model_{timestamp}.zip")
            self.model.save(best_model_path)
            if self.verbose > 0:
                logger.info(f"Best model saved as: {best_model_path}")
        return result

class StopTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StopTrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if stop_training:
            logger.info("Training stopped by user.")
            return False  # Returning False will stop training
        return True

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
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fully connected layers for scalar inputs
        self.scalar_net = nn.Sequential(
            nn.Linear(self.scalar_size, 16),  # Health, hunger, alive
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
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
    global stop_training
    try:
        if hasattr(key, 'char') and key.char == 'i':
            logger.info("Stopping training: 'I' key pressed.")
            stop_training = True
            return False  # Stop listener
    except AttributeError as e:
        logger.error(f"Error in on_press_stop: {e}")

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press_stop)
    listener.start()
    return listener

def on_press_start(key):
    global start_training
    try:
        if hasattr(key, 'char') and key.char == 'o':
            logger.info("Starting training: 'o' key pressed.")
            start_training = True
            return False  # Stop listener
    except AttributeError as e:
        logger.error(f"Error in on_press_start: {e}")

def wait_for_start_key():
    logger.info("Waiting for the 'o' key to start training...")
    try:
        with keyboard.Listener(on_press=on_press_start) as listener:
            listener.join()
    except Exception as e:
        logger.error(f"Exception in wait_for_start_key: {e}")
    logger.info("'o' key pressed. Proceeding with training...")

# -------------------- Model Loading --------------------

def choose_model_to_load(env, policy_kwargs, models_dir, training_params):
    models = [f for f in os.listdir(models_dir) if f.endswith(".zip")]

    if not models:
        logger.info(f"No pre-trained models found in {models_dir}. Starting fresh training.")
        return None

    # Sort models to ensure consistent ordering
    models.sort()
    # Default to the "base default" model (assuming the first one)
    default_model = models[0]

    print("Available models:")
    for idx, model_name in enumerate(models):
        print(f"{idx}: {model_name}")

    choice = input(f"Enter the number of the model you want to load or press Enter to load the default model '{default_model}' or 'n' to start a new model: ")
    if choice.lower() == 'n':
        return None
    elif choice == '':
        model_path = os.path.join(models_dir, default_model)
    else:
        try:
            model_idx = int(choice)
            if model_idx < 0 or model_idx >= len(models):
                print("Invalid selection. Starting fresh training.")
                return None
            model_path = os.path.join(models_dir, models[model_idx])
        except ValueError:
            print("Invalid input. Starting fresh training.")
            return None

    logger.info(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    # Check if the model's observation space matches the environment's
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        model.set_env(env)
        logger.info("Loaded model is compatible with the new observation space.")
    except ValueError:
        logger.warning("Loaded model has incompatible observation space. Reinitializing model.")
        model = DQN(
            policy=CustomDQNPolicy,
            env=env,
            verbose=training_params['verbose'],
            learning_rate=training_params['learning_rate'],
            buffer_size=training_params['buffer_size'],
            learning_starts=training_params['learning_starts'],
            batch_size=training_params['batch_size'],
            gamma=training_params['gamma'],
            train_freq=training_params['train_freq'],
            target_update_interval=training_params['target_update_interval'],
            exploration_fraction=training_params['exploration_fraction'],
            exploration_final_eps=training_params['exploration_final_eps'],
            tensorboard_log=TENSORBOARD_LOG_DIR,
            policy_kwargs=policy_kwargs,
        )
    return model

# -------------------- Custom DQN Policy --------------------

from stable_baselines3.dqn.policies import DQNPolicy

class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs)

# -------------------- Main Training Function --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualization is not implemented in this script")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--scenario", type=str, default="default", help="Select training scenario: default or debug")
    args = parser.parse_args()

    # Update logging level if debug flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        global DEBUG
        DEBUG = True

    training_params = TRAINING_PARAMS
    logger.info("Using DQN training parameters.")

    logger.info("Initializing environment...")
    try:
        # Ensure required directories exist
        for dir_path in [MODELS_DIR, LOGS_DIR, LOGS_EVAL_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Directory checked/created: {dir_path}")

        env = MinecraftEnv(reward_model=REWARD_MODEL)
        env = Monitor(env, filename=LOGS_DIR)
        eval_env = MinecraftEnv(reward_model=REWARD_MODEL)
        eval_env = Monitor(eval_env, filename=LOGS_EVAL_DIR)
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        traceback.print_exc()
        return

    logger.info("Setting policy parameters...")
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[256, 256],  # Increased from [128, 128]
    )

    # Allow user to choose model to load or start fresh
    model = choose_model_to_load(env, policy_kwargs, MODELS_DIR, training_params)

    # If model is None, initialize a new model
    if model is None:
        logger.info("Starting new model...")
        model = DQN(
            policy=CustomDQNPolicy,
            env=env,
            verbose=training_params['verbose'],
            learning_rate=training_params['learning_rate'],
            buffer_size=training_params['buffer_size'],
            learning_starts=training_params['learning_starts'],
            batch_size=training_params['batch_size'],
            gamma=training_params['gamma'],
            train_freq=training_params['train_freq'],
            target_update_interval=training_params['target_update_interval'],
            exploration_fraction=training_params['exploration_fraction'],
            exploration_final_eps=training_params['exploration_final_eps'],
            tensorboard_log=TENSORBOARD_LOG_DIR,
            policy_kwargs=policy_kwargs,
        )

    logger.info("Waiting for 'o' key to start training...")
    wait_for_start_key()

    logger.info("Starting keyboard listener for stopping training...")
    listener = start_keyboard_listener()
    
    checkpoint_callback = TimestampedCheckpointCallback(
        save_freq=400,  # Save model every 400 steps
        save_path=MODELS_DIR,
        verbose=1
    )



    # Callbacks
    eval_callback = TimestampedEvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        save_path=TIMESTAMPED_BEST_MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=5000,  # Adjusted evaluation frequency
        deterministic=True,
        render=False
    )

    stop_training_callback = StopTrainingCallback()

    callback = CallbackList([eval_callback, stop_training_callback, checkpoint_callback])

    total_timesteps = training_params['total_timesteps']

    logger.info("Starting training...")
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=1000, callback=callback)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        traceback.print_exc()

    logger.info("Saving final model...")
    final_model_path = os.path.join(MODELS_DIR, "dqn_minecraft_final")
    model.save(final_model_path)

    logger.info("Closing environments and stopping listener...")
    env.close()
    eval_env.close()
    listener.stop()

    logger.info("Training process has been stopped.")
    sys.exit(0)

if __name__ == "__main__":
    main()

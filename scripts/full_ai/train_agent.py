import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import threading
import argparse
from pynput import keyboard
import traceback
import logging
import time

from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from logging_callback import LoggingCallback  # Ensure this is in your project
from datetime import datetime


from env import MinecraftEnv  # Import your environment

# Set up logging
logger = logging.getLogger(__name__)

# Flags to control training
start_training = False
stop_training = False

def handle_thread_exception(args):
    """Log uncaught exceptions in threads."""
    if issubclass(args.exc_type, KeyboardInterrupt):
        sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)
        return
    logger.error("Uncaught exception in thread:", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

threading.excepthook = handle_thread_exception


class TimestampedCheckpointCallback(BaseCallback):
    """
    A custom callback to save models with a unique timestamp.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(TimestampedCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.save_path, f"ppo_minecraft_{timestamp}.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True





# Define custom feature extractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        self.extractors = {}
        logger.debug("Initializing CustomCombinedExtractor.")

        image_shape = observation_space.spaces['image'].shape
        position_shape = observation_space.spaces['position'].shape
        task_shape = observation_space.spaces['task'].shape[0]  # Task array length

        logger.debug(f"Image shape: {image_shape}")
        logger.debug(f"Position shape: {position_shape}")
        logger.debug(f"Task shape: {task_shape}")

        # Build CNN for image
        n_input_channels = 1  # Grayscale images
        self.image_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_image = torch.zeros((1, n_input_channels, *image_shape))
            n_flatten = self.image_cnn(sample_image).shape[1]
            logger.debug(f"Flattened image feature size: {n_flatten}")

        # Build feedforward network for position
        self.position_net = nn.Sequential(
            nn.Linear(position_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.task_net = nn.Sequential(
            nn.Linear(task_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.scalar_net = nn.Sequential(
            nn.Linear(3, 32),  # Three inputs: health, hunger, alive
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self._features_dim = n_flatten + 64 + 32 + 32  # Image + position + task + scalar
        logger.debug(f"Total features dimension: {self._features_dim}")

    def forward(self, observations):
        logger.debug("Extracting features from observations.")
        # Process image
        image_obs = observations['image'].float()
        if len(image_obs.shape) == 3:  # If no batch dimension
            image_obs = image_obs.unsqueeze(1)  # Add channel dimension
        image_features = self.image_cnn(image_obs)

        # Process position
        position_obs = observations['position'].float()
        position_features = self.position_net(position_obs)

        # Process task
        task_obs = observations['task'].float()
        task_features = self.task_net(task_obs)

        # Process scalar states
        health = observations['health'].view(-1, 1).float()  # Ensure shape [batch_size, 1]
        hunger = observations['hunger'].view(-1, 1).float()
        alive = observations['alive'].view(-1, 1).float()

        scalar_obs = torch.cat([health, hunger, alive], dim=1)
        scalar_features = self.scalar_net(scalar_obs)

        # Concatenate all features
        features = torch.cat((image_features, position_features, task_features, scalar_features), dim=1)
        logger.debug("Features extracted successfully.")
        return features

# Keyboard listener to stop training
def on_press_stop(key):
    global stop_training
    try:
        logger.debug(f"Key pressed: {key}")
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
        logger.debug(f"Key pressed: {key}")
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

# Custom callback to stop training
class StopTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StopTrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if stop_training:
            logger.info("Training stopped by user.")
            return False  # Returning False will stop training
        return True

# Modify environment to remove position setting
class ModifiedMinecraftEnv(MinecraftEnv):
    def __init__(self, *args, action_delay=0.3, **kwargs):
        super(ModifiedMinecraftEnv, self).__init__(*args, **kwargs)
        self.action_delay = action_delay  # Action delay in seconds
        logger.debug("ModifiedMinecraftEnv initialized.")

    def reset(self, seed=None, options=None):
        logger.debug("Environment reset called.")
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        logger.debug(f"Environment step called with action: {action}")
        obs, reward, done, truncated, info = super().step(action)
        logger.debug(f"Step result - Reward: {reward}, Done: {done}")
        return obs, reward, done, truncated, info

def choose_model_to_load(env, policy_kwargs, models_dir, args):
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
        # Load default model
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
    model = PPO.load(model_path)

    # Check if the model's observation space matches the environment's
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        model.set_env(env)
        logger.info("Loaded model is compatible with the new observation space.")
    except ValueError:
        logger.warning("Loaded model has incompatible observation space. Reinitializing model.")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1 if args.debug else 0,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=128,
            n_epochs=20,
            tensorboard_log="./ppo_minecraft_tensorboard/",
            policy_kwargs=policy_kwargs,
        )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Removed visualization from this script")
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
        # Ensure required directories exist
        for dir_path in ['./models/', './logs/', './logs_eval/']:
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Directory checked/created: {dir_path}")

        env = ModifiedMinecraftEnv()
        env = Monitor(env, filename="./logs/")
        eval_env = ModifiedMinecraftEnv()
        eval_env = Monitor(eval_env, filename="./logs_eval/")
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        traceback.print_exc()
        return

    logger.info("Setting policy parameters...")
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # Directory where models are saved
    models_dir = r'C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\models'

    # Allow user to choose model to load or start fresh
    model = choose_model_to_load(env, policy_kwargs, models_dir, args)

    # If model is None, initialize a new model
    if model is None:
        logger.info("Starting new model...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1 if args.debug else 0,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=128,
            n_epochs=20,
            tensorboard_log="./ppo_minecraft_tensorboard/",
            policy_kwargs=policy_kwargs,
        )

    logger.info("Waiting for 'o' key to start training...")
    wait_for_start_key()

    logger.info("Starting keyboard listener for stopping training...")
    listener = start_keyboard_listener()

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=500,
        deterministic=True,
        render=False
    )

    checkpoint_callback = TimestampedCheckpointCallback(
        save_freq=500,  # Save every 500 steps
        save_path=models_dir,  # Directory to save models
        verbose=1  # Print logs when saving
    )
    stop_training_callback = StopTrainingCallback()
    log_dir = r'E:\training data\fullAIlogs'
    logging_callback = LoggingCallback(log_dir=log_dir)

    callback = CallbackList([eval_callback, stop_training_callback, checkpoint_callback, logging_callback])

    # Training variables
    total_timesteps = 100000
    timesteps_per_iteration = 20000
    timesteps_trained = 0
    reset_interval = 20000
    backup_reset_time = 3600
    last_progress_time = time.time()

    def backup_timer():
        nonlocal last_progress_time, model
        try:
            while timesteps_trained < total_timesteps and not stop_training:
                logger.debug("Backup timer is running...")
                time.sleep(60)  # Check every minute
                elapsed_time = time.time() - last_progress_time
                logger.debug(f"Elapsed time since last progress: {elapsed_time:.2f}s")
                if elapsed_time > backup_reset_time:
                    logger.info(f"Backup timer triggered after {elapsed_time / 60:.1f} minutes. Resetting training...")

                    # Find the latest saved model in the models folder
                    if os.path.exists(models_dir):
                        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
                        if model_files:
                            # Sort by modification time to get the latest file
                            latest_model_path = max(
                                [os.path.join(models_dir, f) for f in model_files],
                                key=os.path.getmtime
                            )
                            logger.info(f"Loading the latest model: {latest_model_path}")
                            model = PPO.load(latest_model_path, env=env)
                        else:
                            logger.info("No saved models found. Starting fresh training...")
                            model = PPO(
                                policy="MultiInputPolicy",
                                env=env,
                                verbose=1 if args.debug else 0,
                                learning_rate=0.0003,
                                n_steps=1024,
                                batch_size=128,
                                n_epochs=20,
                                tensorboard_log="./ppo_minecraft_tensorboard/",
                                policy_kwargs=policy_kwargs,
                            )
                    else:
                        logger.info("Models directory does not exist. Starting fresh training...")
                        model = PPO(
                            policy="MultiInputPolicy",
                            env=env,
                            verbose=1 if args.debug else 0,
                            learning_rate=0.0003,
                            n_steps=1024,
                            batch_size=128,
                            n_epochs=20,
                            tensorboard_log="./ppo_minecraft_tensorboard/",
                            policy_kwargs=policy_kwargs,
                        )
                    last_progress_time = time.time()
        except Exception as e:
            logger.error(f"Error in backup_timer: {e}")
            traceback.print_exc()

    logger.info("Starting backup timer thread...")
    backup_thread = threading.Thread(target=backup_timer, daemon=True)
    backup_thread.start()

    logger.info("Starting training loop...")
    try:
        while timesteps_trained < total_timesteps:
            try:
                logger.debug(f"Starting training iteration. Timesteps trained: {timesteps_trained}")
                model.learn(total_timesteps=timesteps_per_iteration, log_interval=1, callback=callback)
                timesteps_trained += timesteps_per_iteration
                logger.debug(f"Training completed for iteration. Total timesteps trained: {timesteps_trained}")
                last_progress_time = time.time()

                if (timesteps_trained+1) % reset_interval == 0:
                    logger.info(f"Reset condition met at timestep {timesteps_trained}. Resetting training...")
                    # Save the current model to a temporary file
                    temp_model_path = os.path.join(models_dir, "temp_model.zip")
                    model.save(temp_model_path)
                    # Reload the model from the saved state
                    model = PPO.load(temp_model_path, env=env)
                    # Optionally, clean up the temporary file
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)

                if stop_training:
                    logger.info("Stop training flag set. Exiting training loop.")
                    break

            except Exception as e:
                logger.error(f"Error during training iteration: {e}")
                traceback.print_exc()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        traceback.print_exc()

    logger.info("Saving final model...")
    final_model_path = os.path.join(models_dir, "ppo_minecraft_final")
    model.save(final_model_path)

    logger.info("Closing environments and stopping listener...")
    env.close()
    eval_env.close()
    listener.stop()

    logger.info("Training process has been stopped.")
    sys.exit(0)

if __name__ == "__main__":
    main()


import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import argparse
from pynput import keyboard
import traceback
import logging
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy

from logging_callback import LoggingCallback
from env import MinecraftEnv  # Import your environment

# -------------------- Configuration --------------------

# Learning rate schedule parameters
INITIAL_LR = 0.0003
DECAY_RATE = 0.99

# Directories
MODELS_DIR = r'C:\Users\odezz\source\MinecraftAI2\scripts\full_ai\models'
LOGS_DIR = "./logs/"
LOGS_EVAL_DIR = "./logs_eval/"
TENSORBOARD_LOG_DIR = "./ppo_minecraft_tensorboard/"
BEST_MODEL_DIR = os.path.join(LOGS_DIR, "best_model")
TIMESTAMPED_BEST_MODELS_DIR = os.path.join(LOGS_DIR, "timestamped_best_models")

# Training parameters (default settings)
TRAINING_PARAMS = {
    'n_steps': 1024,
    'batch_size': 64,
    'n_epochs': 10,
    'total_timesteps': 100000,
    'timesteps_per_iteration': 20000,
    'reset_interval': 20000,
    'backup_reset_time': 3600,  # in seconds
    'learning_rate': INITIAL_LR,
    'decay_rate': DECAY_RATE,
    'verbose': 1  # Set to 1 for verbose output, 0 for minimal output
}

# Alternative training parameters for different scenarios
# Example: Debugging scenario
DEBUG_TRAINING_PARAMS = {
    'n_steps': 256,
    'batch_size': 32,
    'n_epochs': 5,
    'total_timesteps': 50000,
    'timesteps_per_iteration': 10000,
    'reset_interval': 10000,
    'backup_reset_time': 1800,  # in seconds
    'learning_rate': 0.0003,
    'decay_rate': 0.95,
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

# -------------------- Custom Learning Rate Schedule --------------------

class CustomLRSchedule:
    """
    Custom learning rate schedule that starts with a high learning rate and decays it over time.
    """
    def __init__(self, initial_lr=INITIAL_LR, decay_rate=DECAY_RATE):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def __call__(self, progress_remaining):
        return self.initial_lr * (self.decay_rate ** (1 - progress_remaining))

# lr_schedule will be initialized later based on selected training parameters

# -------------------- Custom Callbacks --------------------

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

class LogLrCallback(BaseCallback):
    """
    Custom callback for logging the learning rate during training.
    """
    def __init__(self, verbose=0):
        super(LogLrCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if 'lr_scheduler' in self.locals:
            lr = self.locals['lr_scheduler']._last_lr[0]
            self.logger.record("train/learning_rate", lr)
            if self.verbose > 0:
                logger.debug(f"Learning rate: {lr}")
        return True

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.save_path, f"ppo_minecraft_{timestamp}.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                logger.info(f"Saving model checkpoint to {model_path}")
        return True

class StopTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StopTrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if stop_training:
            logger.info("Training stopped by user.")
            return False  # Returning False will stop training
        return True


# -------------------- Custom Feature Extractor --------------------

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # Extract shapes from observation space
        image_shape = observation_space.spaces['image'].shape
        position_shape = observation_space.spaces['position'].shape
        task_shape = observation_space.spaces['task'].shape[0]

        # CNN for image input
        n_input_channels = 1  # Assuming grayscale images
        self.image_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size of image features
        with torch.no_grad():
            sample_image = torch.zeros((1, n_input_channels, *image_shape))
            self.image_feature_dim = self.image_cnn(sample_image).view(-1).shape[0]

        # Shared network for combined position and task
        self.position_task_net = nn.Sequential(
            nn.Linear(position_shape[0] + task_shape, 128),  # Combine position and task
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Output compressed size
            nn.ReLU(),
        )

        # Separate network for other scalar inputs
        self.scalar_net = nn.Sequential(
            nn.Linear(3, 32),  # Health, hunger, alive
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Compute total feature dimension
        self._features_dim = self.image_feature_dim + 64 + 32
        logger.info(f"CustomCombinedExtractor initialized with features_dim={self._features_dim}")

    def forward(self, observations):
        # Process image input
        image_obs = observations['image'].float()
        logger.debug(f"Original image_obs shape: {image_obs.shape}")

        # Ensure image_obs is [batch_size, channels, height, width]
        if len(image_obs.shape) == 2:
            image_obs = image_obs.unsqueeze(0).unsqueeze(0)
        elif len(image_obs.shape) == 3:
            image_obs = image_obs.unsqueeze(1)

        logger.debug(f"Processed image_obs shape: {image_obs.shape}")
        image_features = self.image_cnn(image_obs)
        logger.debug(f"Image features shape: {image_features.shape}")

        # Combine position and task features
        batch_size = image_obs.shape[0]
        position_obs = observations['position'].float().view(batch_size, -1)
        task_obs = observations['task'].float().view(batch_size, -1)
        position_task_input = torch.cat((position_obs, task_obs), dim=1)
        position_task_features = self.position_task_net(position_task_input)

        # Process scalar inputs
        health = observations['health'].view(batch_size, 1).float()
        hunger = observations['hunger'].view(batch_size, 1).float()
        alive = observations['alive'].view(batch_size, 1).float()
        scalar_obs = torch.cat([health, hunger, alive], dim=1)
        scalar_features = self.scalar_net(scalar_obs)

        # Concatenate all features
        features = torch.cat((image_features, position_task_features, scalar_features), dim=1)
        logger.debug(f"Concatenated features shape: {features.shape}")

        return features

# -------------------- Custom Policy --------------------

class MaskedPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MaskedPolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Policy
        action_logits = self.action_net(latent_pi)

        # Handle missing action_mask
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            # Apply the mask to the logits.
            masked_logits = action_logits + (1 - action_mask) * -1e9
        else:
            # Backup: Create a custom action_mask if missing.
            # Assuming we have 6 possible actions (indexed 0 through 5)
            custom_action_mask = np.zeros(action_logits.shape)  # All actions invalid by default
            custom_action_mask[[0, 23, 24]] = 1  # Set actions 0, 23, and 24 as valid
            # Apply the custom action_mask to the logits
            masked_logits = action_logits + (1 - custom_action_mask) * -1e9


        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)
        actions = distribution.get_actions(deterministic=deterministic)

        # Value function
        values = self.value_net(latent_vf)

        return actions, values, distribution.log_prob(actions)

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
    model = PPO.load(model_path)

    # Check if the model's observation space matches the environment's
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        model.set_env(env)
        logger.info("Loaded model is compatible with the new observation space.")
    except ValueError:
        logger.warning("Loaded model has incompatible observation space. Reinitializing model.")
        model = PPO(
            policy=MaskedPolicy,
            env=env,
            verbose=training_params['verbose'],
            learning_rate=CustomLRSchedule(
                initial_lr=training_params['learning_rate'],
                decay_rate=training_params['decay_rate']
            ),
            n_steps=training_params['n_steps'],
            batch_size=training_params['batch_size'],
            n_epochs=training_params['n_epochs'],
            tensorboard_log=TENSORBOARD_LOG_DIR,
            policy_kwargs=policy_kwargs,
        )
    return model

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

    # Select training parameters based on scenario
    if args.scenario == "debug":
        training_params = DEBUG_TRAINING_PARAMS
        logger.info("Using DEBUG training parameters.")
    else:
        training_params = TRAINING_PARAMS
        logger.info("Using DEFAULT training parameters.")

    # Initialize the learning rate schedule with selected parameters
    lr_schedule = CustomLRSchedule(
        initial_lr=training_params['learning_rate'],
        decay_rate=training_params['decay_rate']
    )

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
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Allow user to choose model to load or start fresh
    model = choose_model_to_load(env, policy_kwargs, MODELS_DIR, training_params)

    # If model is None, initialize a new model
    if model is None:
        logger.info("Starting new model...")
        model = PPO(
            policy=MaskedPolicy,
            env=env,
            verbose=training_params['verbose'],
            learning_rate=lr_schedule,
            n_steps=training_params['n_steps'],
            batch_size=training_params['batch_size'],
            n_epochs=training_params['n_epochs'],
            tensorboard_log=TENSORBOARD_LOG_DIR,
            policy_kwargs=policy_kwargs,
        )

    logger.info("Waiting for 'o' key to start training...")
    wait_for_start_key()

    logger.info("Starting keyboard listener for stopping training...")
    listener = start_keyboard_listener()

    # Callbacks
    log_lr_callback = LogLrCallback()
    eval_callback = TimestampedEvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        save_path=TIMESTAMPED_BEST_MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=500,
        deterministic=True,
        render=False
    )

    checkpoint_callback = TimestampedCheckpointCallback(
        save_freq=400,
        save_path=MODELS_DIR,
        verbose=1
    )
    stop_training_callback = StopTrainingCallback()
    logging_callback = LoggingCallback(log_dir=r'E:\training data\fullAIlogs')

    callback = CallbackList([eval_callback, stop_training_callback, checkpoint_callback, logging_callback, log_lr_callback])

    # Training variables
    total_timesteps = training_params['total_timesteps']
    timesteps_per_iteration = training_params['timesteps_per_iteration']
    timesteps_trained = 0
    backup_reset_time = training_params['backup_reset_time']
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
                    if os.path.exists(MODELS_DIR):
                        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.zip')]
                        if model_files:
                            latest_model_path = max(
                                [os.path.join(MODELS_DIR, f) for f in model_files],
                                key=os.path.getmtime
                            )
                            logger.info(f"Loading the latest model: {latest_model_path}")
                            model = PPO.load(latest_model_path, env=env)
                        else:
                            logger.info("No saved models found. Starting fresh training...")
                            model = PPO(
                                policy=MaskedPolicy,
                                env=env,
                                verbose=training_params['verbose'],
                                learning_rate=lr_schedule,
                                n_steps=training_params['n_steps'],
                                batch_size=training_params['batch_size'],
                                n_epochs=training_params['n_epochs'],
                                tensorboard_log=TENSORBOARD_LOG_DIR,
                                policy_kwargs=policy_kwargs,
                            )
                    else:
                        logger.info("Models directory does not exist. Starting fresh training...")
                        model = PPO(
                            policy=MaskedPolicy,
                            env=env,
                            verbose=training_params['verbose'],
                            learning_rate=lr_schedule,
                            n_steps=training_params['n_steps'],
                            batch_size=training_params['batch_size'],
                            n_epochs=training_params['n_epochs'],
                            tensorboard_log=TENSORBOARD_LOG_DIR,
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

                # Load the best model after evaluation
                best_model_path = os.path.join(eval_callback.best_model_save_path, "best_model.zip")
                if os.path.exists(best_model_path):
                    logger.info("Using the best model for the next training iteration.")
                    model = PPO.load(best_model_path, env=env)
                else:
                    logger.info("No best model found. Continuing with the current model.")

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
    final_model_path = os.path.join(MODELS_DIR, "ppo_minecraft_final")
    model.save(final_model_path)

    logger.info("Closing environments and stopping listener...")
    env.close()
    eval_env.close()
    listener.stop()

    logger.info("Training process has been stopped.")
    sys.exit(0)

if __name__ == "__main__":
    main()

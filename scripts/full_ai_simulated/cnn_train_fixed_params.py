import os
import logging
import datetime
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer


from cnn_env import SimulatedEnvGraphics  # Import the environment
import signal
from torch.utils.tensorboard import SummaryWriter

#PrioritizedReplayBuffer

# Configuration and Hyperparameters

# Paths and Directories
MODEL_PATH = r"E:\CNN"  # Path to save and load the models
LOG_DIR = r"C:\Users\odezz\source\MinecraftAI2\scripts\full_ai_simulated\simulated_tensorboard"
LOG_FILE = r"E:\CNN\training_data.csv"

# General Parameters
PARALLEL_ENVS = 1  # Number of parallel environments
RENDER_MODE = "none"
SAVE_EVERY_STEPS = 50000  # Save the model every N steps
TOTAL_TIMESTEPS = 5_000_000  # Total timesteps for training
ADDITIONAL_TIMESTEPS = 1_000_000  # Additional timesteps for continued training

# Training Parameters
LEARNING_RATE = 0.001
BUFFER_SIZE = 40_000  # Increased buffer size
BATCH_SIZE = 128  # Increased batch size
GAMMA = 0.91
TRAIN_FREQ = 4  # Increased train frequency
GRADIENT_STEPS = 1  # Number of gradient steps per update
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.05
EXPLORATION_FINAL_EPS = 0.01

# CNN Parameters
FEATURES_DIM = 256  # Output dimensions of the CNN feature extractor
CNN_FILTERS = [32, 64, 64]  # Filters for each convolutional layer
CNN_KERNEL_SIZES = [8, 4, 3]  # Kernel sizes for each convolutional layer
CNN_STRIDES = [4, 2, 1]  # Strides for each convolutional layer

# Environment Parameters
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 1  # Grayscale image
GRID_SIZE = 2000
CELL_SIZE = 50
TASK_SIZE = 20
MAX_EPISODE_LENGTH = 500
SIMULATION_SPEED = 5
ZOOM_FACTOR = 0.2
DEVICE = "cpu"  # Default device for environment

# Agent Parameters
INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
YAW_RANGE = (-180, 180)
PITCH_RANGE = (-90, 90)
POSITION_RANGE = (-120, 120)

# Action Constants
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
YAW_CHANGE = 10  # Degrees to turn left or right
ACTION_SPACE_SIZE = 25  # Placeholder for future use

# Reward Parameters
REWARD_SCALE_POSITIVE = 10
REWARD_SCALE_NEGATIVE = 5
REWARD_PENALTY_STAY_STILL = -3
REWARD_MAX = 10
REWARD_MIN = -10

# Evaluation Parameters
EVAL_FREQUENCY = 20_000  # Evaluate every N steps
MOVING_AVG_WINDOW = 10  # Use smoothed rewards over N episodes

# Miscellaneous
VERBOSE = 1  # Verbosity level

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    """
    def func(progress_remaining):
        return initial_value * progress_remaining
    return func


class TensorboardLoggingCallback(BaseCallback):
    """
    Custom callback for logging rewards, episode info, and action distributions to TensorBoard.
    """
    def __init__(self, log_dir, verbose=VERBOSE):
        super(TensorboardLoggingCallback, self).__init__(verbose)
        self.writer = None
        self.log_dir = log_dir  # Base directory for logs

    def _on_training_start(self) -> None:
        # Generate a unique folder for this training session
        unique_dir = os.path.join(
            self.log_dir,
            f"DQN_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(unique_dir, exist_ok=True)

        # Initialize TensorBoard writer in the unique directory
        self.writer = SummaryWriter(log_dir=unique_dir)
        self.start_time = datetime.datetime.now()

    def _on_step(self) -> bool:
        # Log only every `log_interval` steps
        if self.num_timesteps % 100 == 0:
            # Log rewards
            if "rewards" in self.locals:
                rewards = self.locals["rewards"]  # Extract rewards from self.locals
                if rewards is not None:
                    mean_reward = np.mean(rewards)
                    self.writer.add_scalar("Reward/Step Reward", mean_reward, self.num_timesteps)

            # Log additional episode information
            if "infos" in self.locals:
                infos = self.locals["infos"]
                for info in infos:
                    if "episode" in info.keys():
                        episode_reward = info["episode"]["r"]
                        self.writer.add_scalar("Reward/Episode Reward", episode_reward, self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        # Close the TensorBoard writer
        self.writer.close()


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback to save the model every N steps.
    """
    def __init__(self, save_freq, save_path, verbose=VERBOSE):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        timesteps = self.num_timesteps
        if timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_step_{timesteps}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved at step {timesteps} to {save_file}")
        return True


def make_env(seed, render_mode=RENDER_MODE, enable_logging=True):
    """
    Create a simulated environment instance.
    """
    def _init():
        env = SimulatedEnvGraphics(
            render_mode=render_mode,
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            task_size=TASK_SIZE,
            max_episode_length=MAX_EPISODE_LENGTH,
            simulation_speed=SIMULATION_SPEED,
            zoom_factor=ZOOM_FACTOR,
            device=DEVICE,
            log_file=LOG_FILE if enable_logging else None,
            enable_logging=enable_logging,
        )
        env.seed(seed)
        return env
    return _init


class SaveBestModelOnEvalCallback(BaseCallback):
    """
    Custom callback to save the best model during evaluation with smoothed rewards.
    Includes functionality to save the last model when training ends or upon a manual stop.
    """
    def __init__(self, eval_env, save_path, verbose=VERBOSE, eval_frequency=EVAL_FREQUENCY, moving_avg_window=MOVING_AVG_WINDOW):
        super(SaveBestModelOnEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.best_smoothed_reward = -float("inf")
        self.eval_frequency = eval_frequency
        self.moving_avg_window = moving_avg_window
        self.reward_history = []  # Keep track of recent rewards for moving average
        self.stop_training = False  # Signal to stop training
        os.makedirs(save_path, exist_ok=True)

        # Handle interrupt signal to stop training
        signal.signal(signal.SIGINT, self._handle_stop_signal)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_frequency == 0:
            mean_reward = self.evaluate_model()
            self.reward_history.append(mean_reward)

            # Maintain the reward history window size
            if len(self.reward_history) > self.moving_avg_window:
                self.reward_history.pop(0)

            # Compute smoothed reward
            smoothed_reward = np.mean(self.reward_history)
            logger.info(f"Evaluation completed: Mean Reward={mean_reward:.2f}, Smoothed Reward={smoothed_reward:.2f}")

            # Save the model if the smoothed reward is the best so far
            if smoothed_reward > self.best_smoothed_reward:
                self.best_smoothed_reward = smoothed_reward
                self.save_best_model(smoothed_reward)

        # Stop training if stop signal is triggered
        if self.stop_training:
            logger.info("Manual stop triggered. Saving the last model...")
            self.save_last_model()
            return False  # Stop training

        return True

    def evaluate_model(self):
        logger.info("Starting evaluation...")
        total_rewards = []

        for episode in range(self.moving_avg_window):
            obs = self.eval_env.reset()
            #print("Raw observation from reset:", obs)

            # Unwrap observation if coming from VecEnv
            if isinstance(obs, (list, tuple)):
                obs = obs[0]

            if not isinstance(obs, dict) or "image" not in obs or "other" not in obs:
                raise ValueError(f"Invalid observation format at reset: {obs}")

            done = False
            episode_reward = 0

            while not done:
                obs_preprocessed = {
                    "image": np.array(obs["image"], dtype=np.float32),
                    "other": np.array(obs["other"], dtype=np.float32),
                }
                action, _ = self.model.predict(obs_preprocessed, deterministic=True)
                
                # Correct unpacking of values
                obs, reward, done, info = self.eval_env.step(action)

                # Unwrap observation again if necessary
                if isinstance(obs, (list, tuple)):
                    obs = obs[0]

                episode_reward += reward

            total_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}: Reward = {episode_reward}")

        mean_reward = np.mean(total_rewards)
        return mean_reward

    def save_best_model(self, smoothed_reward):
        """
        Save the model with a timestamp if it achieves a new best smoothed reward.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_file = os.path.join(self.save_path, f"best_model_{timestamp}.zip")
        self.model.save(save_file)
        logger.info(f"New best model saved at {save_file} with smoothed reward: {smoothed_reward:.2f}")

    def save_last_model(self):
        """
        Save the current model when training is manually stopped or ends.
        """
        save_file = os.path.join(self.save_path, "last_model.zip")
        self.model.save(save_file)
        logger.info(f"Last model saved at {save_file}")

    def _handle_stop_signal(self, signum, frame):
        """
        Handle SIGINT (Ctrl+C) signal to stop training gracefully.
        """
        logger.info("Received stop signal (Ctrl+C). Preparing to save the last model and stop training...")
        self.stop_training = True


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observation spaces, combining CNN for image and MLP for other data.
    """
    def __init__(
        self,
        observation_space,
        features_dim=FEATURES_DIM,
        cnn_filters=CNN_FILTERS,
        cnn_kernel_sizes=CNN_KERNEL_SIZES,
        cnn_strides=CNN_STRIDES,
        mlp_hidden_sizes=[128, 128],
        device="cuda",
        image_weight=0.1,
    ):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        # Set the device
        self.device = th.device(device)
        self.image_weight = image_weight
        # Initialize extractors
        self.extractors = {}

        # Validate observation space keys
        assert "image" in observation_space.spaces, "Observation space must contain 'image' key."
        assert "other" in observation_space.spaces, "Observation space must contain 'other' key."

        # Build the CNN for the 'image' key
        image_shape = observation_space["image"].shape
        cnn_layers = []
        input_channels = image_shape[0]

        # Dynamically create CNN layers
        for i in range(len(cnn_filters)):
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=cnn_filters[i],
                    kernel_size=cnn_kernel_sizes[i],
                    stride=cnn_strides[i],
                )
            )
            cnn_layers.append(nn.ReLU())
            input_channels = cnn_filters[i]
        cnn_layers.append(nn.Flatten())

        self.extractors["image"] = nn.Sequential(*cnn_layers).to(self.device)  # Move CNN to device

        # Compute the output size of the CNN dynamically
        with th.no_grad():
            sample_input = th.zeros(1, *image_shape, device=self.device)
            cnn_output_size = self.extractors["image"](sample_input).shape[1]

        # Build the MLP for the 'other' key
        other_shape = observation_space["other"].shape[0]
        mlp_layers = []
        input_size = other_shape

        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            input_size = hidden_size

        self.extractors["other"] = nn.Sequential(*mlp_layers).to(self.device)  # Move MLP to device

        # Calculate total combined features dimension
        self._features_dim = cnn_output_size + input_size

    def forward(self, observations):
        """
        Forward pass to extract features from 'image' and 'other' observations,
        with learned emphasis on task and position vectors.
        """
        # Ensure observations contain expected keys
        assert "image" in observations, "Input observations must contain 'image' key."
        assert "other" in observations, "Input observations must contain 'other' key."

        # Extract image features
        image_features = self.extractors["image"](observations["image"].to(self.device))

        # Split the 'other' vector into components
        other_features = observations["other"].to(self.device)
        position_vector = other_features[:, :5]  # Position features
        task_vector = other_features[:, 5:25]  # Task features
        other_scalar_features = other_features[:, 25:]  # Other scalar features

        # Pass each component through individual small MLPs (learned weighting)
        position_processed = nn.Linear(5, 5).to(self.device)(position_vector)
        task_processed = nn.Linear(20, 20).to(self.device)(task_vector)
        other_scalar_processed = nn.Linear(other_scalar_features.size(1), other_scalar_features.size(1)).to(self.device)(
            other_scalar_features
        )

        # Concatenate all processed parts
        weighted_other_features = th.cat([position_processed, task_processed, other_scalar_processed], dim=1)

        # Pass the combined 'other' features through the main MLP
        other_processed = self.extractors["other"](weighted_other_features)

        # Combine image and processed 'other' features
        image_features = image_features * self.image_weight
        combined_features = th.cat((image_features, other_processed), dim=1)

        return combined_features


def create_model(env):
    """
    Create the DQN model with the specified parameters.
    """
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            features_dim=FEATURES_DIM,
            cnn_filters=CNN_FILTERS,
            cnn_kernel_sizes=CNN_KERNEL_SIZES,
            cnn_strides=CNN_STRIDES,
            mlp_hidden_sizes=[128, 128],
            image_weight=0.1,
            device="cuda" if th.cuda.is_available() else "cpu",
        ),
    )

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=linear_schedule(LEARNING_RATE),
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        policy_kwargs=policy_kwargs,
        verbose=VERBOSE,
        tensorboard_log=LOG_DIR,
        device="cuda" if th.cuda.is_available() else "cpu",
    )
    return model


def create_callbacks(model_path):
    """
    Create callbacks for saving models and logging.
    """
    save_on_step_callback = SaveOnStepCallback(
        save_freq=SAVE_EVERY_STEPS,
        save_path=model_path,
        verbose=VERBOSE
    )

    save_best_model_callback = SaveBestModelOnEvalCallback(
        eval_env=DummyVecEnv([make_env(seed=0, render_mode="none")]),
        save_path=model_path,
        verbose=VERBOSE,
        eval_frequency=EVAL_FREQUENCY,
        moving_avg_window=MOVING_AVG_WINDOW
    )

    tensorboard_callback = TensorboardLoggingCallback(
        log_dir=LOG_DIR,
        verbose=VERBOSE
    )

    return [save_on_step_callback, save_best_model_callback, tensorboard_callback]


def train_agent(env, model_path, total_timesteps, model=None):
    """
    Train the agent using pre-defined hyperparameters on GPU and save the best models.
    """
    os.makedirs(model_path, exist_ok=True)

    if model is None:
        model = create_model(env)

    callbacks = create_callbacks(model_path)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )

    model.save(f"{model_path}/final_model")
    env.close()


def main():
    print("Starting the program...")
    print(f"Using device: {'cuda' if th.cuda.is_available() else 'cpu'}")

    # Check for existing models
    existing_models = [
        file for file in os.listdir(MODEL_PATH) if file.endswith(".zip")
    ]

    if existing_models:
        print("Existing models found:")
        for idx, model_name in enumerate(existing_models):
            print(f"{idx + 1}. {model_name}")
        print(f"{len(existing_models) + 1}. Train a new model")

        # User selection
        while True:
            try:
                choice = int(input("Enter the number of your choice: "))
                if 1 <= choice <= len(existing_models) + 1:
                    break
                else:
                    print(f"Please select a number between 1 and {len(existing_models) + 1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Load or train based on user selection
        if choice == len(existing_models) + 1:
            print("Training a new model...")
            env = DummyVecEnv([make_env(seed=i) for i in range(PARALLEL_ENVS)])
            train_agent(env, MODEL_PATH, total_timesteps=TOTAL_TIMESTEPS)
        else:
            selected_model = os.path.join(MODEL_PATH, existing_models[choice - 1])
            print(f"Loading model: {selected_model}")

            # Recreate the environment
            env = DummyVecEnv([make_env(seed=i) for i in range(PARALLEL_ENVS)])

            # Reinitialize the model with current parameters
            model = create_model(env)

            # Load the saved model
            print("Loading saved model parameters...")
            model.load(selected_model, device="cuda" if th.cuda.is_available() else "cpu")
            print("Model loaded successfully. Ready for evaluation or further training.")

            # Continue training
            print(f"Continuing training for {ADDITIONAL_TIMESTEPS} timesteps...")

            # Continue training with callbacks
            train_agent(env, MODEL_PATH, total_timesteps=ADDITIONAL_TIMESTEPS, model=model)

            print("Training continued successfully.")
    else:
        print("No existing models found. Training a new model...")
        env = DummyVecEnv([make_env(seed=i) for i in range(PARALLEL_ENVS)])
        train_agent(env, MODEL_PATH, total_timesteps=TOTAL_TIMESTEPS)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)
        import traceback
        traceback.print_exc()

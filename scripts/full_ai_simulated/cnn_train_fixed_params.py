import os
import logging
import datetime
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from cnn_env import SimulatedEnvGraphics  # Import the environment
import signal


# Hardcoded paths and parameters
MODEL_PATH = r"E:\CNN"  # Path to save and load the models
PARALLEL_ENVS = 1  # Number of parallel environments
RENDER_MODE = "none"
SAVE_EVERY_STEPS = 50000  # Save the model every 10,000 steps
TOTAL_TIMESTEPS = 5000000  # Total timesteps for training

# Training parameters
LEARNING_RATE = 0.0002
BUFFER_SIZE = 10000
BATCH_SIZE = 224
GAMMA = 0.91
TRAIN_FREQ = 4
TARGET_UPDATE_INTERVAL = 1500
EXPLORATION_FRACTION = 0.4
EXPLORATION_FINAL_EPS = 0.05
EVAL_FREQ = 5000

# CNN parameters
FEATURES_DIM = 256  # Output dimensions of the CNN feature extractor
CNN_FILTERS = [32, 64, 64]  # Filters for each convolutional layer
CNN_KERNEL_SIZES = [8, 4, 3]  # Kernel sizes for each convolutional layer
CNN_STRIDES = [4, 2, 1]  # Strides for each convolutional layer

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback to save the model every N steps.
    """
    def __init__(self, save_freq, save_path, verbose=1):
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


def make_env(render_mode="none"):
    """
    Create a simulated environment instance.
    """
    def _init():
        return SimulatedEnvGraphics(render_mode=render_mode)
    return _init


class SaveBestModelOnEvalCallback(BaseCallback):
    """
    Custom callback to save the best model during evaluation with smoothed rewards.
    Includes functionality to save the last model when training ends or upon a manual stop.
    """
    def __init__(self, eval_env, save_path, verbose=1, eval_frequency=1000, moving_avg_window=25):
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
        """
        Evaluate the current model using the provided evaluation environment.
        """
        logger.info("Starting evaluation...")
        total_rewards = []

        for episode in range(self.moving_avg_window):  # Evaluate over `moving_avg_window` episodes
            obs = self.eval_env.reset()
            done = [False] * self.eval_env.num_envs
            episode_reward = 0

            while not all(done):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                episode_reward += rewards[0]
                done = dones

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
        features_dim=256,  # Default combined feature size
        cnn_filters=[32, 64, 64],
        cnn_kernel_sizes=[8, 4, 3],
        cnn_strides=[4, 2, 1],
        mlp_hidden_sizes=[256, 256],
        device="cuda",  # Default to GPU
    ):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)

        # Set the device
        self.device = th.device(device)

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
        Forward pass to extract features from 'image' and 'other' observations.
        """
        # Ensure observations contain expected keys
        assert "image" in observations, "Input observations must contain 'image' key."
        assert "other" in observations, "Input observations must contain 'other' key."

        # Extract features and move to the correct device
        image_features = self.extractors["image"](observations["image"].to(self.device))
        other_features = self.extractors["other"](observations["other"].to(self.device))

        # Concatenate features
        combined_features = th.cat((image_features, other_features), dim=1)
        return combined_features





def train_agent(env, model_path, total_timesteps):
    """
    Train the agent using pre-defined hyperparameters on GPU and save the best models.
    """
    os.makedirs(model_path, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            features_dim=FEATURES_DIM,
            cnn_filters=CNN_FILTERS,
            cnn_kernel_sizes=CNN_KERNEL_SIZES,
            cnn_strides=CNN_STRIDES,
            mlp_hidden_sizes=[256, 256],
            device="cuda" if th.cuda.is_available() else "cpu",  # Explicitly pass GPU device
        ),
    )

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./simulated_tensorboard/",
        device="cuda" if th.cuda.is_available() else "cpu",  # Enable GPU
    )

    save_on_step_callback = SaveOnStepCallback(
        save_freq=SAVE_EVERY_STEPS,
        save_path=model_path,
        verbose=1
    )

    save_best_model_callback = SaveBestModelOnEvalCallback(
        eval_env=DummyVecEnv([make_env(render_mode="none")]),
        save_path=MODEL_PATH,
        verbose=1,
        eval_frequency=4000,  # Evaluate every 1000 steps
        moving_avg_window=10  # Use smoothed rewards over 25 episodes
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[save_on_step_callback, save_best_model_callback]
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
            env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])
            train_agent(env, MODEL_PATH, total_timesteps=TOTAL_TIMESTEPS)
        else:
            selected_model = os.path.join(MODEL_PATH, existing_models[choice - 1])
            print(f"Loading model: {selected_model}")

            # Recreate the environment
            env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])

            # Load the saved model
            print("Reinitializing model with current parameters and loading saved weights...")
            saved_model = DQN.load(selected_model, device="cuda" if th.cuda.is_available() else "cpu")

            # Reinitialize the model with current parameters
            policy_kwargs = dict(
                features_extractor_class=CustomCombinedExtractor,
                features_extractor_kwargs=dict(
                    features_dim=FEATURES_DIM,
                    cnn_filters=CNN_FILTERS,
                    cnn_kernel_sizes=CNN_KERNEL_SIZES,
                    cnn_strides=CNN_STRIDES,
                    mlp_hidden_sizes=[256, 256],
                    device="cuda" if th.cuda.is_available() else "cpu",  # Explicitly pass GPU device
                ),
            )

            model = DQN(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=LEARNING_RATE,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                train_freq=TRAIN_FREQ,
                target_update_interval=TARGET_UPDATE_INTERVAL,
                exploration_fraction=EXPLORATION_FRACTION,
                exploration_final_eps=EXPLORATION_FINAL_EPS,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./simulated_tensorboard/",
                device="cuda" if th.cuda.is_available() else "cpu",
            )

            # Load parameters from the saved model into the new model
            model.set_parameters(saved_model.get_parameters())
            print("Model loaded successfully. Ready for evaluation or further training.")

            # Continue training
            ADDITIONAL_TIMESTEPS = 500000  # Adjust as needed
            print(f"Continuing training for {ADDITIONAL_TIMESTEPS} timesteps...")

            # Define callbacks
            save_on_step_callback = SaveOnStepCallback(
                save_freq=SAVE_EVERY_STEPS,
                save_path=MODEL_PATH,
                verbose=1
            )

            save_best_model_callback = SaveBestModelOnEvalCallback(
                eval_env=DummyVecEnv([make_env(render_mode="none")]),
                save_path=MODEL_PATH,
                verbose=1,
                eval_frequency=4000,
                moving_avg_window=10
            )

            # Continue training with callbacks
            model.learn(
                total_timesteps=ADDITIONAL_TIMESTEPS,
                callback=[save_on_step_callback, save_best_model_callback]
            )

            print("Training continued successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)
        import traceback
        traceback.print_exc()
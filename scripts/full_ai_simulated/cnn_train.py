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
import optuna
import csv

# Hardcoded paths and parameters
MODEL_PATH = r"E:\CNN"  # Path to save and load the models
PARALLEL_ENVS = 1  # Number of parallel environments
RENDER_MODE = "none"
SAVE_EVERY_STEPS = 50000  # Save the model every 50,000 steps
TOTAL_TIMESTEPS = 10000  # Timesteps for training during Optuna optimization
ADDITIONAL_TIMESTEPS = 500000  # Timesteps for additional training
LOG_FILE = r"E:\CNN\optuna_trials_log.csv"

# CNN parameters
FEATURES_DIM = 256  # Output dimensions of the CNN feature extractor
CNN_FILTERS = [32, 64, 64]  # Filters for each convolutional layer
CNN_KERNEL_SIZES = [8, 4, 3]  # Kernel sizes for each convolutional layer
CNN_STRIDES = [4, 2, 1]  # Strides for each convolutional layer


# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def initialize_logging():
    """
    Initialize the log file and write the header if it doesn't exist.
    """
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Header for the log file
            writer.writerow(["trial_number", "learning_rate", "buffer_size", "batch_size", 
                             "gamma", "exploration_fraction", "exploration_final_eps", 
                             "mean_reward"])

def log_trial(trial_number, parameters, mean_reward):
    """
    Log trial parameters and results to the log file.
    """
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial_number,
            parameters["learning_rate"],
            parameters["buffer_size"],
            parameters["batch_size"],
            parameters["gamma"],
            parameters["exploration_fraction"],
            parameters["exploration_final_eps"],
            mean_reward
        ])



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



def optimize_hyperparameters(trial):
    """
    Optuna trial function to optimize hyperparameters.
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)  # Log scale
    buffer_size = trial.suggest_int("buffer_size", 10000, 40000, step=5000)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 224, 256])
    gamma = trial.suggest_float("gamma", 0.89, 0.99)  # Linear scale
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.3)  # Linear scale
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.04, 0.08)  # Linear scale
    train_freq = trial.suggest_int("train_freq", 1, 4)
    target_update_interval = trial.suggest_int("target_update_interval", 500, 2000, step=500)  # 500 to 3000 in steps of 500

    
    # Environment
    env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])

    # Policy configuration
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            features_dim=FEATURES_DIM,
            cnn_filters=CNN_FILTERS,
            cnn_kernel_sizes=CNN_KERNEL_SIZES,
            cnn_strides=CNN_STRIDES,
            mlp_hidden_sizes=[256, 256],
            device="cuda" if th.cuda.is_available() else "cpu",
        ),
    )

    # Model
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log="./simulated_tensorboard/",
        device="cuda" if th.cuda.is_available() else "cpu",
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Evaluate the trained model
    eval_rewards = []
    for _ in range(10):  # Evaluate over 10 episodes
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, _ = env.step(action)
            episode_reward += rewards
        eval_rewards.append(episode_reward)

    mean_reward = np.mean(eval_rewards)
    env.close()

    # Log trial results
    parameters = {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "train_freq": train_freq,
        "target_update_interval": target_update_interval,
    }
    log_trial(trial.number, parameters, mean_reward)

    return mean_reward




def main():
    # Check for existing models

    initialize_logging()

    existing_models = [
        file for file in os.listdir(MODEL_PATH) if file.endswith(".zip")
    ]

    if existing_models:
        print("Existing models found:")
        for idx, model_name in enumerate(existing_models):
            print(f"{idx + 1}. {model_name}")
        print(f"{len(existing_models) + 1}. Train a new model or optimize hyperparameters with Optuna")

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

        if choice == len(existing_models) + 1:
            # Run Optuna hyperparameter optimization
            print("Starting Optuna optimization...")
            study = optuna.create_study(direction="maximize")
            study.optimize(optimize_hyperparameters, n_trials=40)
            print("Best trial:")
            print(study.best_trial)
            print("Best hyperparameters:")
            for key, value in study.best_params.items():
                print(f"{key}: {value}")
        else:
            selected_model = os.path.join(MODEL_PATH, existing_models[choice - 1])
            print(f"Loading model: {selected_model}")

            # Recreate the environment
            env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])

            # Load the saved model
            print("Reinitializing model with current parameters and loading saved weights...")
            saved_model = DQN.load(selected_model, device="cuda" if th.cuda.is_available() else "cpu")

            # Continue training
            print(f"Continuing training for {ADDITIONAL_TIMESTEPS} timesteps...")
            saved_model.learn(total_timesteps=ADDITIONAL_TIMESTEPS)
            print("Training continued successfully.")
    else:
        print("No existing models found. Starting Optuna optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_hyperparameters, n_trials=10)
        print("Best trial:")
        print(study.best_trial)
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:", e)
        import traceback
        traceback.print_exc()
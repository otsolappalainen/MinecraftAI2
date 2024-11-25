# sim_train.py

import os
import datetime
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn as nn

# Suppress TensorFlow logs if TensorFlow is not needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO logs

# Import your environment
from sim_env import SimulatedEnvSimplified  # Updated simplified environment

# Configuration and Hyperparameters
MODEL_PATH_SIMPLIFIED_TRANSFER = "models_simplified_transfer"
MODEL_PATH_FULL = "models_full"
LOG_DIR = "tensorboard_logs"
LOG_FILE = "training_data.csv"

# Ensure directories exist
os.makedirs(MODEL_PATH_SIMPLIFIED_TRANSFER, exist_ok=True)
os.makedirs(MODEL_PATH_FULL, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# General Parameters
PARALLEL_ENVS = 8  # Reduced from 8 to 4 to manage memory
TOTAL_TIMESTEPS = 25_000_000

# Training Parameters
LOW_LR = 1e-4
HIGH_LR = 1e-4
BUFFER_SIZE = 30_000  # Reduced from 50_000
BATCH_SIZE = 64  # Reduced from 128
GAMMA = 0.95
TRAIN_FREQ = 4
GRADIENT_STEPS = 2
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.4
EXPLORATION_FINAL_EPS = 0.05
SAVE_EVERY_STEPS = 500_000
EVAL_FREQ = 2_000
EVAL_EPISODES = 5
VERBOSE = 1

# Callbacks
class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=VERBOSE):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_step_{self.num_timesteps}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved at step {self.num_timesteps} to {save_file}")
        return True

class TimestampedEvalCallback(EvalCallback):
    """
    Custom EvalCallback to save the best model only if it's better than the current best.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf  # Initialize to a very low value

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Check if a new best model was found
        if self.last_mean_reward is not None and self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            if self.best_model_save_path is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.best_model_save_path, f"best_model_{timestamp}.zip")
                self.model.save(save_path)
                print(f"New best model saved with mean reward {self.best_mean_reward:.2f} at {save_path}")
        return result


class FullModelFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(FullModelFeatureExtractor, self).__init__(observation_space, features_dim=128)

        # Updated to handle the new `other` size (28 elements)
        scalar_input_dim = observation_space["other"].shape[0]
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        image_shape = observation_space["image"].shape
        self.image_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = self._get_conv_output_size(image_shape)
        self.fusion_layers = nn.Sequential(
            nn.Linear(64 + conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def _get_conv_output_size(self, shape):
        dummy_input = th.zeros(1, *shape)
        output = self.image_net(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, observations):
        scalar_obs = observations["other"]
        scalar_features = self.scalar_net(scalar_obs)
        
        image_obs = observations["image"]
        image_features = self.image_net(image_obs)
        
        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        return fused_features  # Only return fused features

def make_env_simplified(env_id, rank, seed=0):
    def _init():
        env = SimulatedEnvSimplified(
            log_file=LOG_FILE,
            enable_logging=True,
            env_id=rank
        )
        env.seed(seed + rank)  # Now works as seed method is implemented
        return env
    return _init

def custom_learning_rate_schedule(initial_lr_low, lr_high, total_timesteps, low_pct=0.5, high_pct=0.8):
    low_timesteps = int(low_pct * total_timesteps)
    high_timesteps = int(high_pct * total_timesteps)

    def lr_schedule(progress_remaining):
        current_timestep = (1 - progress_remaining) * total_timesteps
        if current_timestep <= low_timesteps:
            return initial_lr_low
        elif current_timestep <= low_timesteps + high_timesteps:
            return lr_high
        else:
            decay_progress = (current_timestep - (low_timesteps + high_timesteps)) / (
                total_timesteps - (low_timesteps + high_timesteps)
            )
            return initial_lr_low + (lr_high - initial_lr_low) * (1 - decay_progress)

    return lr_schedule


def transfer_weights(simplified_model, full_model):
    """
    Transfer weights from the simplified model to the full model, skipping mismatched layers.
    """
    simplified_state_dict = simplified_model.policy.state_dict()
    full_state_dict = full_model.policy.state_dict()

    # Iterate over simplified model's state_dict and match with the full model
    for key, param in simplified_state_dict.items():
        if key in full_state_dict:
            if full_state_dict[key].shape == param.shape:
                # Transfer weights if shapes match
                full_state_dict[key] = param
                print(f"Transferred: {key}")
            else:
                # Skip mismatched layers
                print(f"Skipped (shape mismatch): {key} ({param.shape} vs {full_state_dict[key].shape})")

    # Load the updated state_dict into the full model
    full_model.policy.load_state_dict(full_state_dict)
    print("Weight transfer completed.")


def create_simplified_model(env):
    lr_schedule = custom_learning_rate_schedule(LOW_LR, HIGH_LR, TOTAL_TIMESTEPS)
    policy_kwargs = dict(
        features_extractor_class=FullModelFeatureExtractor,
        net_arch=[],  # Can be adjusted as needed
    )
    return DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=lr_schedule,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        verbose=VERBOSE,
        tensorboard_log=LOG_DIR,
        device="cuda" if th.cuda.is_available() else "cpu",
        policy_kwargs=policy_kwargs,
    )

def main():
    print("Select an option:")
    print("1. Load weights from simplified model and transfer to full model")
    print("2. Load an existing full model")
    print("3. Start training the full model from scratch")
    choice = input("Enter 1, 2, or 3: ").strip()

    env = SubprocVecEnv([make_env_simplified(i, i) for i in range(PARALLEL_ENVS)])

    if choice == "1":
        # Define the expected simplified model filename
        expected_model_name = "simplified_model.zip"
        simplified_model_path = os.path.join(MODEL_PATH_SIMPLIFIED_TRANSFER, expected_model_name)

        if not os.path.exists(simplified_model_path):
            # List available models for clarity
            available_models = [f for f in os.listdir(MODEL_PATH_SIMPLIFIED_TRANSFER) if f.endswith(".zip")]
            if not available_models:
                print(f"No simplified models found in '{MODEL_PATH_SIMPLIFIED_TRANSFER}'.")
                return
            else:
                print(f"Available models in '{MODEL_PATH_SIMPLIFIED_TRANSFER}':")
                for idx, model in enumerate(available_models, start=1):
                    print(f"{idx}. {model}")
                # Prompt user to select a model
                while True:
                    try:
                        selection = int(input(f"Enter the number of the model to transfer from (1-{len(available_models)}): "))
                        if 1 <= selection <= len(available_models):
                            selected_model = available_models[selection - 1]
                            simplified_model_path = os.path.join(MODEL_PATH_SIMPLIFIED_TRANSFER, selected_model)
                            break
                        else:
                            print(f"Please enter a number between 1 and {len(available_models)}.")
                    except ValueError:
                        print("Invalid input. Please enter a valid number.")
        else:
            print(f"Found '{expected_model_name}' in '{MODEL_PATH_SIMPLIFIED_TRANSFER}'.")

        print(f"Loading simplified model from '{simplified_model_path}'...")
        simplified_model = DQN.load(simplified_model_path, env=None)  # Load without env
        full_model = create_simplified_model(env)
        print("Transferring weights...")
        transfer_weights(simplified_model, full_model)

    elif choice == "2":
        model_files = [f for f in os.listdir(MODEL_PATH_FULL) if f.endswith(".zip")]
        if not model_files:
            print(f"No existing models found in '{MODEL_PATH_FULL}'.")
            env.close()
            return

        print("Available full models:")
        for idx, model_file in enumerate(model_files, start=1):
            print(f"{idx}. {model_file}")

        while True:
            try:
                model_choice = int(input(f"Select the model to load (1-{len(model_files)}): ")) - 1
                if 0 <= model_choice < len(model_files):
                    selected_model_path = os.path.join(MODEL_PATH_FULL, model_files[model_choice])
                    break
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"Loading full model from '{selected_model_path}'...")
        full_model = DQN.load(selected_model_path, env=env)
        print("Full model loaded successfully.")

    elif choice == "3":
        print("Starting training from scratch...")
        full_model = create_simplified_model(env)

    else:
        print("Invalid choice. Exiting.")
        return

    # Create evaluation environment
    eval_env = SubprocVecEnv([make_env_simplified(0, 0)])

    # Define callbacks
    callbacks = [
        SaveOnStepCallback(save_freq=SAVE_EVERY_STEPS, save_path=MODEL_PATH_FULL),
        TimestampedEvalCallback(
            eval_env=eval_env,
            best_model_save_path=MODEL_PATH_FULL,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
        ),
    ]

    # Start training
    full_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    final_model_path = os.path.join(MODEL_PATH_FULL, "final_model.zip")
    full_model.save(final_model_path)
    print(f"Training completed. Model saved to '{final_model_path}'.")

if __name__ == "__main__":
    main()

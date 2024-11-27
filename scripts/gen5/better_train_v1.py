import os
import datetime
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import multiprocessing
import logging
import logging.handlers
import queue  # Correctly import the queue module

# Suppress TensorFlow logs if TensorFlow is not needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO logs

# Import your environment
from better_env import SimplifiedEnvWithBlankImage  # Updated simplified environment

# Configuration and Hyperparameters
MODEL_PATH_OLD = "models_new"
MODEL_PATH_NEW = "models_full_simple"
LOG_DIR = "tensorboard_logs_new_v1"
LOG_FILE = "training_data.csv"

# Ensure directories exist
os.makedirs(MODEL_PATH_OLD, exist_ok=True)
os.makedirs(MODEL_PATH_NEW, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Automatically set PARALLEL_ENVS based on available CPU cores
CPU_COUNT = multiprocessing.cpu_count()
PARALLEL_ENVS = 16  # Adjusted from 16 to 8 to match available CPU cores
print(f"Using {PARALLEL_ENVS} parallel environments out of {CPU_COUNT} available CPU cores.")

# General Parameters
TOTAL_TIMESTEPS = 20_000_000

# Training Parameters
LOW_LR = 1e-5
HIGH_LR = 1e-4
BUFFER_SIZE = 10_000  # Further reduced from 10,000
BATCH_SIZE = 64  # Reduced from 128
GAMMA = 0.95
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1000  # Adjusted from 4000
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.05
SAVE_EVERY_STEPS = 500_000
EVAL_FREQ = 3_000
EVAL_EPISODES = 20
VERBOSE = 1

# Configurable Modes
ENABLE_MIXED_PRECISION = True  # Set to False to disable mixed precision
ADJUST_NETWORK_ARCHITECTURE = False  # Set to False to always use CNN

# Device Configuration
device = th.device("cuda" if th.cuda.is_available() else "cpu")
if device.type == 'cuda':
    th.cuda.set_device(0)  # Ensure GPU 0 is used
    th.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking for speed

# Setup asynchronous logging
logger = logging.getLogger('env_logger')
logger.setLevel(logging.INFO)
log_queue = queue.Queue()  # Use queue.Queue() instead of logging.handlers.Queue()
handler = logging.handlers.QueueHandler(log_queue)
logger.addHandler(handler)
listener = logging.handlers.QueueListener(log_queue, logging.FileHandler(LOG_FILE))
listener.start()

# Callbacks
class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=VERBOSE):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.last_save_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            save_file = os.path.join(self.save_path, f"model_step_{self.num_timesteps}.zip")
            self.model.save(save_file)
            self.last_save_step = self.num_timesteps
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
        # Initialize with a placeholder for features_dim; it will be overwritten
        super(FullModelFeatureExtractor, self).__init__(observation_space, features_dim=256)

        # Scalar observation processing
        scalar_input_dim = observation_space["other"].shape[0]
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        scalar_output_size = 128

        # Always use CNN layers for image processing
        image_input_channels = observation_space["image"].shape[0]  # Assuming 3 for RGB images
        self.image_net = nn.Sequential(
            nn.Conv2d(image_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Precompute the CNN output size
        dummy_image = th.zeros(1, *observation_space["image"].shape)
        conv_output_size = self._get_conv_output_size(dummy_image)
        self.conv_output_size = conv_output_size  # Store for debugging

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(scalar_output_size + conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._features_dim = 128

        print("Using CNN layers for feature extractor.")

    def _get_conv_output_size(self, dummy_input):
        """
        Calculate the output size of the CNN given an input shape.
        """
        with th.no_grad():
            output = self.image_net(dummy_input)
            return int(np.prod(output.shape[1:]))

    def forward(self, observations):
        # Process scalar inputs
        scalar_features = self.scalar_net(observations["other"])

        # Process image inputs through CNN
        image_features = self.image_net(observations["image"])

        # Concatenate scalar and image features
        combined_input = th.cat([scalar_features, image_features], dim=1)

        # Pass through fusion layers
        fused_features = self.fusion_layers(combined_input)

        return fused_features

def make_env_simplified(env_id, rank, seed=0):
    def _init():
        env = SimplifiedEnvWithBlankImage(
            log_file=LOG_FILE,
            enable_logging=True,
            env_id=rank
        )
        env.seed(seed + rank)  # Now works as seed method is implemented
        return env
    return _init

def custom_learning_rate_schedule(initial_lr_low, lr_high, total_timesteps, low_pct=0.02, high_pct=0.5):
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

def transfer_weights(old_model, new_model):
    """
    Transfer weights from the old model to the new model, skipping mismatched layers.
    """
    old_state_dict = old_model.policy.state_dict()
    new_state_dict = new_model.policy.state_dict()

    transferred_layers = []
    skipped_layers = []

    # Iterate over old model's state_dict and match with the new model
    for key, param in old_state_dict.items():
        if key in new_state_dict:
            if new_state_dict[key].shape == param.shape:
                # Transfer weights if shapes match
                new_state_dict[key] = param
                transferred_layers.append(key)
            else:
                # Skip mismatched layers
                skipped_layers.append((key, param.shape, new_state_dict[key].shape))
        else:
            skipped_layers.append((key, param.shape, 'Not present in new model'))

    # Load the updated state_dict into the new model
    new_model.policy.load_state_dict(new_state_dict)
    print("Weight transfer completed.")
    print(f"Transferred layers ({len(transferred_layers)}):")
    for layer in transferred_layers:
        print(f"  - {layer}")
    print(f"Skipped layers ({len(skipped_layers)}):")
    for layer, old_shape, new_shape in skipped_layers:
        print(f"  - {layer}: {old_shape} -> {new_shape}")

def validate_transfer(old_model, new_model):
    """
    Validate that the weights have been transferred correctly.
    """
    for key in old_model.policy.state_dict():
        if key in new_model.policy.state_dict():
            if new_model.policy.state_dict()[key].equal(old_model.policy.state_dict()[key]):
                print(f"Validation passed for layer: {key}")
            else:
                print(f"Validation failed for layer: {key}")

def create_simplified_model(env):
    lr_schedule = custom_learning_rate_schedule(LOW_LR, HIGH_LR, TOTAL_TIMESTEPS)
    policy_kwargs = dict(
        features_extractor_class=FullModelFeatureExtractor,
        net_arch=[64, 64],  # Decision layers with 64 neurons each
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
        device=device,
        policy_kwargs=policy_kwargs,
        optimize_memory_usage=False,  # Disabled to avoid AssertionError
    )

def main():
    print("Select an option:")
    print("1. Load weights from old model and transfer to new model")
    print("2. Load an existing new model")
    print("3. Start training the new model from scratch")
    choice = input("Enter 1, 2, or 3: ").strip()

    # Create training environment
    env_fns = [make_env_simplified(i, i) for i in range(PARALLEL_ENVS)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    if choice == "1":
        # Load weights from old model and transfer
        model_files = [f for f in os.listdir(MODEL_PATH_OLD) if f.endswith(".zip")]
        if not model_files:
            print(f"No existing models found in '{MODEL_PATH_OLD}'.")
            env.close()
            listener.stop()
            return

        print("Available old models:")
        for idx, model_file in enumerate(model_files, start=1):
            print(f"{idx}. {model_file}")

        while True:
            try:
                model_choice = int(input(f"Select the old model to load (1-{len(model_files)}): ")) - 1
                if 0 <= model_choice < len(model_files):
                    selected_model_path = os.path.join(MODEL_PATH_OLD, model_files[model_choice])
                    break
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"Loading old model from '{selected_model_path}'...")
        old_model = DQN.load(selected_model_path, env=None)  # Load model without environment

        # Create a new model with current parameters
        new_model = create_simplified_model(env)
        print("Transferring weights from the old model to match current parameters...")
        transfer_weights(old_model, new_model)

        # Optional: Validate the transfer
        validate_transfer(old_model, new_model)

    elif choice == "2":
        # Load an existing new model
        model_files = [f for f in os.listdir(MODEL_PATH_NEW) if f.endswith(".zip")]
        if not model_files:
            print(f"No existing models found in '{MODEL_PATH_NEW}'.")
            env.close()
            listener.stop()
            return

        print("Available new models:")
        for idx, model_file in enumerate(model_files, start=1):
            print(f"{idx}. {model_file}")

        while True:
            try:
                model_choice = int(input(f"Select the model to load (1-{len(model_files)}): ")) - 1
                if 0 <= model_choice < len(model_files):
                    selected_model_path = os.path.join(MODEL_PATH_NEW, model_files[model_choice])
                    break
                else:
                    print("Invalid choice. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print(f"Loading new model from '{selected_model_path}'...")
        old_model = DQN.load(selected_model_path, env=None)  # Load model without environment

        # Create a new model with current parameters
        new_model = create_simplified_model(env)
        print("Transferring weights from the loaded model to match current parameters...")
        transfer_weights(old_model, new_model)

        # Optional: Validate the transfer
        validate_transfer(old_model, new_model)

    elif choice == "3":
        # Start training from scratch
        print("Starting training from scratch...")
        new_model = create_simplified_model(env)

    else:
        print("Invalid choice. Exiting.")
        env.close()
        listener.stop()
        return

    # Create evaluation environment
    eval_env_fn = make_env_simplified(0, 0)
    eval_env = SubprocVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)

    # Define callbacks
    callbacks = [
        SaveOnStepCallback(save_freq=SAVE_EVERY_STEPS, save_path=MODEL_PATH_NEW),
        TimestampedEvalCallback(
            eval_env=eval_env,
            best_model_save_path=MODEL_PATH_NEW,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
        ),
    ]

    # Start training with mixed precision if enabled
    if ENABLE_MIXED_PRECISION and device.type == 'cuda':
        th.set_float32_matmul_precision('medium')
        print("Mixed precision training enabled.")

    try:
        # Start training
        new_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
        final_model_path = os.path.join(MODEL_PATH_NEW, "final_model.zip")
        new_model.save(final_model_path)
        print(f"Training completed. Model saved to '{final_model_path}'.")
    except th.cuda.OutOfMemoryError:
        print("CUDA Out of Memory error encountered. Reducing batch size and retrying...")
        th.cuda.empty_cache()
        # Optionally, implement a retry mechanism with reduced parameters
    finally:
        # Ensure proper closure
        env.close()
        eval_env.close()
        listener.stop()

if __name__ == "__main__":
    main()

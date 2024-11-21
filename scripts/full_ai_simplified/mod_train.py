import os
import logging
import datetime
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces  # Import spaces from gymnasium
import signal
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from env import SimulatedEnvSimplified  # Import the simplified environment

# Configuration and Hyperparameters

# Paths and Directories
MODEL_PATH = "models"  # Path to save and load the models
LOG_DIR = "tensorboard_logs"
LOG_FILE = "training_data.csv"

# General Parameters
PARALLEL_ENVS = 8  # Number of parallel environments
RENDER_MODE = "none"
SAVE_EVERY_STEPS = 5_000_000  # Save the model every N steps
TOTAL_TIMESTEPS = 50_000_000  # Total timesteps for training

# Training Parameters
LEARNING_RATE = 0.008
BUFFER_SIZE = 50_000  # Buffer size
BATCH_SIZE = 128  # Batch size
GAMMA = 0.95
TRAIN_FREQ = 4  # Train every 4 steps
GRADIENT_STEPS = 2  # Number of gradient steps per update
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.4
EXPLORATION_FINAL_EPS = 0.05
LOW_LR = 1e-4
HIGH_LR = 1e-3

# Evaluation Parameters
EVAL_FREQ = 10_000  # Evaluate every N steps
EVAL_EPISODES = 25  # Number of episodes for evaluation
MOVING_AVG_WINDOW = 10  # Moving average window for rewards

# Miscellaneous
VERBOSE = 1  # Verbosity level

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = SimulatedEnvSimplified(
            render_mode=RENDER_MODE,
            log_file=LOG_FILE,
            enable_logging=True,
            env_id=rank
        )
        env.seed(seed + rank)
        return env
    return _init



def custom_learning_rate_schedule(initial_lr_low, lr_high, total_timesteps, low_pct=0.05, high_pct=0.2):
    """
    Custom learning rate schedule.
    """
    low_timesteps = int(low_pct * total_timesteps)
    high_timesteps = int(high_pct * total_timesteps)
    
    def lr_schedule(progress_remaining):
        # Calculate current timestep
        current_timestep = (1 - progress_remaining) * total_timesteps

        # Phase 1: Low learning rate
        if current_timestep <= low_timesteps:
            return initial_lr_low
        
        # Phase 2: High learning rate
        elif current_timestep <= low_timesteps + high_timesteps:
            return lr_high
        
        # Phase 3: Gradual decrease back to low
        else:
            # Progress within the decay phase
            decay_progress = (current_timestep - (low_timesteps + high_timesteps)) / (total_timesteps - (low_timesteps + high_timesteps))
            return initial_lr_low + (lr_high - initial_lr_low) * (1 - decay_progress)

    return lr_schedule

class TimestampedEvalCallback(EvalCallback):
    """
    Custom EvalCallback to save the best model with a unique timestamp.
    """
    def _on_step(self) -> bool:
        # Call the original method to retain evaluation behavior
        result = super()._on_step()

        # Check if a new best model was found and best_model_save_path is set
        if self.best_model_save_path is not None and self.n_calls % self.eval_freq == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.best_model_save_path, f"best_model_{timestamp}.zip")
            self.model.save(save_path)
            logger.info(f"New best model saved with timestamp: {save_path}")
        return result

class SaveOnStepCallback(BaseCallback):
    """
    Custom callback for saving a model every N steps.
    """
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
                logger.info(f"Model saved at step {self.num_timesteps} to {save_file}")
        return True

# Custom Feature Extractor
class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes scalar inputs, includes placeholders for image features,
    and passes through fusion and decision layers.
    """
    def __init__(self, observation_space: spaces.Box):
        # The output of the feature extractor will be features_dim (final output of decision layers)
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=24)  # Output: Q-values for actions

        # Scalar input processing (64+64)
        self.scalar_input_dim = observation_space.shape[0]  # Should be 5
        self.scalar_net = nn.Sequential(
            nn.Linear(self.scalar_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Placeholder for future image features
        self.image_placeholder_dim = 64  # e.g., 64
        self.image_placeholder = nn.Parameter(th.zeros(self.image_placeholder_dim), requires_grad=False)

        # Fusion layers (combine scalar features and image placeholder)
        self.fusion_layers = nn.Sequential(
            nn.Linear(64 + self.image_placeholder_dim, 128),  # Combining scalar and image features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Decision layers (output Q-values for 24 actions)
        self.decision_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 24),  # 24 possible actions
        )

    def forward(self, observations):
        # Process scalar inputs
        scalar_features = self.scalar_net(observations)

        # Expand the image placeholder to match batch size
        batch_size = observations.shape[0]
        image_features = self.image_placeholder.unsqueeze(0).expand(batch_size, -1).to(observations.device)

        # Concatenate scalar features and image features
        fused_input = th.cat([scalar_features, image_features], dim=1)

        # Fusion processing
        fused_features = self.fusion_layers(fused_input)

        # Decision-making
        q_values = self.decision_layers(fused_features)
        return q_values


def create_model(env):
    """
    Create the DQN model with specified parameters and custom feature extractor.
    """
    initial_lr_low = LOW_LR
    lr_high = HIGH_LR

    lr_schedule = custom_learning_rate_schedule(
        initial_lr_low=initial_lr_low,
        lr_high=lr_high,
        total_timesteps=TOTAL_TIMESTEPS,
        low_pct=0.05,  # First 5% low
        high_pct=0.2   # Next 20% high
    )

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        net_arch=[128, 128],  # Decision layers: 128+128 neurons
    )

    model = DQN(
        policy="MlpPolicy",  # Using MLP policy with custom feature extractor
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
    return model

def main():
    print("Starting the simplified training script...")
    
    print(f"Using device: {'cuda' if th.cuda.is_available() else 'cpu'}")

    # Create the directory for models
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Create the vectorized training environment
    env = SubprocVecEnv([make_env('SimulatedEnvSimplified', i) for i in range(PARALLEL_ENVS)])

    # Create the DQN model
    model = create_model(env)

    # Create custom callbacks
    callbacks = [SaveOnStepCallback(save_freq=SAVE_EVERY_STEPS, save_path=MODEL_PATH)]

    # Create the evaluation environment using SubprocVecEnv with n=1
    def make_eval_env():
        return SimulatedEnvSimplified(render_mode=RENDER_MODE, log_file=LOG_FILE, enable_logging=False)

    eval_env = SubprocVecEnv([make_eval_env])  # n=1 for evaluation

    # Create the TimestampedEvalCallback
    eval_callback = TimestampedEvalCallback(
        eval_env=eval_env,
        callback_on_new_best=None,
        best_model_save_path=MODEL_PATH,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)

    # Handle graceful shutdown on interrupt
    def handle_interrupt(signum, frame):
        logger.info("Interrupt received, stopping training...")
        model.save(os.path.join(MODEL_PATH, "interrupted_model.zip"))
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    print(model.policy.features_extractor)

    # Start training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks
    )

    # Save the final model
    model.save(os.path.join(MODEL_PATH, "final_model.zip"))
    logger.info("Training completed and model saved.")

    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("An error occurred during training:", exc_info=True)

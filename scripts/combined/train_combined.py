import os
import logging
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import spaces
import signal
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# Import your environments
from simplified_env import SimulatedEnvSimplified  # Simplified environment
from env_combined import SimulatedEnvGraphics  # Full environment with image inputs





# Configuration and Hyperparameters
MODEL_PATH_SIMPLIFIED = "models_simplified"
MODEL_PATH_FULL = "models_full"
LOG_DIR = "tensorboard_logs"
LOG_FILE = "training_data.csv"

# Ensure directories exist
os.makedirs(MODEL_PATH_SIMPLIFIED, exist_ok=True)
os.makedirs(MODEL_PATH_FULL, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# General Parameters
PARALLEL_ENVS = 8
TOTAL_TIMESTEPS = 10_000_000  # Adjust as needed

# Training Parameters
LEARNING_RATE = 0.0005
BUFFER_SIZE = 50_000
BATCH_SIZE = 128
GAMMA = 0.95
TRAIN_FREQ = 4
GRADIENT_STEPS = 2
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.4
EXPLORATION_FINAL_EPS = 0.05
VERBOSE = 1

# Define your make_env functions
def make_env_simplified(env_id, rank, seed=0):
    def _init():
        env = SimulatedEnvSimplified(
            render_mode="none",
            log_file=LOG_FILE,
            enable_logging=True,
            env_id=rank
        )
        env.seed(seed + rank)
        return env
    return _init

def make_env_full(env_id, rank, seed=0):
    def _init():
        env = SimulatedEnvGraphics(
            render_mode="none",
            log_file=LOG_FILE,
            enable_logging=True,
            env_id=rank
        )
        env.seed(seed + rank)
        return env
    return _init



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



class FullModelFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that processes scalar inputs and image inputs,
    combines them using fusion layers, and outputs features for decision layers.
    """
    def __init__(self, observation_space):
        # Assuming observation_space is a Dict space with keys 'scalar' and 'image'
        super(FullModelFeatureExtractor, self).__init__(observation_space, features_dim=24)  # Output size matches decision layers

        # Scalar input processing (same as simplified model)
        scalar_input_dim = observation_space['scalar'].shape[0]
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Image input processing (new in full model)
        image_shape = observation_space['image'].shape
        self.image_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._get_conv_output_size(image_shape), 64),  # Adjust output size
            nn.ReLU(),
        )

        # Fusion layers (same as simplified model)
        self.fusion_layers = nn.Sequential(
            nn.Linear(64 + 64, 128),  # Combining scalar and image features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Decision layers (same as simplified model)
        self.decision_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 24),  # 24 possible actions
        )

    def _get_conv_output_size(self, shape):
        """
        Utility function to compute the output size of the convolutional layers.
        """
        o = th.zeros(1, *shape)
        o = self.image_net[:-3](o)  # Exclude the last Linear and ReLU layers
        return int(np.prod(o.size()))

    def forward(self, observations):
        # Process scalar inputs
        scalar_obs = observations['scalar']
        scalar_features = self.scalar_net(scalar_obs)

        # Process image inputs
        image_obs = observations['image']
        image_features = self.image_net(image_obs)

        # Concatenate scalar and image features
        fused_input = th.cat([scalar_features, image_features], dim=1)

        # Fusion processing
        fused_features = self.fusion_layers(fused_input)

        # Decision-making
        q_values = self.decision_layers(fused_features)
        return q_values


def transfer_weights(simplified_model, full_model):
    """
    Transfer weights from the simplified model to the full model.
    """
    simplified_state_dict = simplified_model.policy.state_dict()
    full_state_dict = full_model.policy.state_dict()

    # Transfer scalar_net weights
    for key in simplified_state_dict:
        if 'features_extractor.scalar_net' in key:
            # Adjust the key for the full model
            full_key = key.replace('features_extractor.', 'features_extractor.')
            if full_key in full_state_dict:
                full_state_dict[full_key] = simplified_state_dict[key]

    # Transfer fusion_layers weights
    for key in simplified_state_dict:
        if 'features_extractor.fusion_layers' in key:
            full_key = key.replace('features_extractor.', 'features_extractor.')
            if full_key in full_state_dict:
                full_state_dict[full_key] = simplified_state_dict[key]

    # Transfer decision_layers weights
    for key in simplified_state_dict:
        if 'features_extractor.decision_layers' in key:
            full_key = key.replace('features_extractor.', 'features_extractor.')
            if full_key in full_state_dict:
                full_state_dict[full_key] = simplified_state_dict[key]

    # Load the updated state_dict into the full model
    full_model.policy.load_state_dict(full_state_dict)


def create_simplified_model(env):
    """
    Create the simplified DQN model.
    """
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        net_arch=[128, 128],
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
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

def create_full_model(env):
    """
    Create the full DQN model.
    """
    policy_kwargs = dict(
        features_extractor_class=FullModelFeatureExtractor,
        net_arch=[],
    )

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
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
    # Train Simplified Model if not already trained
    simplified_model_path = os.path.join(MODEL_PATH_SIMPLIFIED, "final_model.zip")
    if not os.path.exists(simplified_model_path):
        print("Training simplified model...")
        env_simplified = SubprocVecEnv([make_env_simplified('SimulatedEnvSimplified', i) for i in range(PARALLEL_ENVS)])
        simplified_model = create_simplified_model(env_simplified)
        simplified_model.learn(total_timesteps=TOTAL_TIMESTEPS)
        simplified_model.save(simplified_model_path)
        env_simplified.close()
        print("Simplified model trained and saved.")
    else:
        print("Loading simplified model...")
        simplified_model = DQN.load(simplified_model_path)
        print("Simplified model loaded.")

    # Create the full environment
    env_full = SubprocVecEnv([make_env_full('SimulatedEnvGraphics', i) for i in range(PARALLEL_ENVS)])
    
    # Create the full model
    full_model = create_full_model(env_full)
    
    # Transfer weights
    transfer_weights(simplified_model, full_model)

    transferred_model_path = os.path.join(MODEL_PATH_FULL, "transferred_model.zip")

    if os.path.exists(transferred_model_path):
        print("Loading transferred full model...")
        full_model = DQN.load(transferred_model_path)
    else:
        print("Transferring weights from simplified model...")
        transfer_weights(simplified_model, full_model)
        full_model.save(transferred_model_path)
        print("Transferred weights saved to:", transferred_model_path)




    print("Weights transferred from simplified model to full model.")
    
    # Train the full model
    full_model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # Save the full model
    full_model.save(os.path.join(MODEL_PATH_FULL, "final_full_model.zip"))
    env_full.close()
    print("Full model trained and saved.")

if __name__ == "__main__":
    main()

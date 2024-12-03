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
from dqn_env import MinecraftEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from stable_baselines3.common.torch_layers import create_mlp

# Configuration
MODEL_PATH_BC = "models_bc"  # Path to BC models
MODEL_PATH_DQN = "models_dqn_from_bc"
LOG_DIR = "tensorboard_logs_dqn_from_bc"
PARALLEL_ENVS = 1

# Hyperparameters - matching BC where possible
TOTAL_TIMESTEPS = 200_000
LEARNING_RATE = 2e-6  # Same as BC
BUFFER_SIZE = 3_000
BATCH_SIZE = 128  # Same as BC
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 500
EXPLORATION_FRACTION = 0.001
EXPLORATION_FINAL_EPS = 0.2
EVAL_FREQ = 1500
EVAL_EPISODES = 1

# Add save frequency constant
SAVE_EVERY_STEPS = 5_000  # Save every 100k steps

# Ensure directories exist
os.makedirs(MODEL_PATH_DQN, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Device configuration
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Add at top of file
DEBUG_MODE = True  # Toggle for verbose logging

def debug_log(msg):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

class BCMatchingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        debug_log("Starting BCMatchingFeatureExtractor initialization")
        debug_log(f"Observation space during feature extractor initialization: {observation_space}")
        
        # Initialize with placeholder features_dim
        super(BCMatchingFeatureExtractor, self).__init__(observation_space, features_dim=256)
        debug_log("Parent constructor called successfully")
        
        # Calculate input dimensions
        scalar_input_dim = observation_space["other"].shape[0] + observation_space["task"].shape[0]
        image_input_channels = observation_space["image"].shape[0]
        debug_log(f"Input dimensions calculated - scalar: {scalar_input_dim}, image: {image_input_channels}")
        
        try:
            # Initialize networks
            debug_log("Initializing scalar network...")
            self.scalar_net = nn.Sequential(
                nn.Linear(scalar_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3)
            )
            
            debug_log("Initializing image network...")
            self.image_net = nn.Sequential(
                nn.Conv2d(image_input_channels, 32, kernel_size=8, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout2d(p=0.3),
                nn.Flatten()
            )
            
            # Calculate CNN output size
            dummy_image = th.zeros(1, *observation_space["image"].shape)
            conv_output_size = self._get_conv_output_size(dummy_image)
            debug_log(f"CNN output size: {conv_output_size}")
            
            self.fusion_layers = nn.Sequential(
                nn.Linear(128 + conv_output_size, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3)
            )
            
            # Set final features dim after all layers are initialized
            self._features_dim = 128
            debug_log(f"Features dim set to {self._features_dim}")
            
            # Test forward pass
            debug_log("Testing forward pass...")
            dummy_obs = {
                "image": dummy_image,
                "other": th.zeros(1, observation_space["other"].shape[0]),
                "task": th.zeros(1, observation_space["task"].shape[0])
            }
            with th.no_grad():
                test_out = self.forward(dummy_obs)
                debug_log(f"Forward pass successful - output shape: {test_out.shape}")
                
        except Exception as e:
            debug_log(f"Error during feature extractor initialization: {e}")
            raise e  # Re-raise to prevent silent failures

    def _get_conv_output_size(self, dummy_input):
        with th.no_grad():
            output = self.image_net(dummy_input)
            return int(np.prod(output.shape[1:]))

    def forward(self, observations):
        """Process input through feature extractor networks"""
        try:
            # Combine other and task observations
            other_combined = th.cat([observations["other"], observations["task"]], dim=1)
            
            # Process scalar and image features
            scalar_features = self.scalar_net(other_combined)
            image_features = self.image_net(observations["image"])
            
            # Combine and process through fusion layers
            combined = th.cat([scalar_features, image_features], dim=1)
            features = self.fusion_layers(combined)
            
            return features
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Observation shapes: {[(k, v.shape) for k, v in observations.items()]}")
            raise e

# Comment out or remove the CustomDQNPolicy class
# class CustomDQNPolicy(DQNPolicy):
#     ...

def transfer_bc_weights_to_dqn(bc_model_path, dqn_model):
    try:
        print(f"Loading BC model from {bc_model_path}")
        bc_state_dict = th.load(bc_model_path, map_location=device, weights_only=True)
        
        # Create and attach feature extractor
        feature_extractor = BCMatchingFeatureExtractor(dqn_model.observation_space)
        feature_extractor = feature_extractor.to(device)
        dqn_model.policy.features_extractor = feature_extractor
        dqn_fe_state_dict = feature_extractor.state_dict()
        
        transferred_layers = []
        skipped_layers = []
        
        # Try to transfer each layer
        for dqn_key in dqn_fe_state_dict.keys():
            bc_key = dqn_key
            
            if bc_key in bc_state_dict:
                if bc_state_dict[bc_key].shape == dqn_fe_state_dict[dqn_key].shape:
                    # Direct transfer for matching shapes
                    dqn_fe_state_dict[dqn_key].copy_(bc_state_dict[bc_key])
                    transferred_layers.append(dqn_key)
                else:
                    # Special handling for mismatched scalar_net.0.weight
                    if dqn_key == "scalar_net.0.weight":
                        bc_weight = bc_state_dict[bc_key]
                        dqn_weight = dqn_fe_state_dict[dqn_key]
                        
                        # Transfer first 48 input features
                        min_features = min(bc_weight.shape[1], dqn_weight.shape[1])
                        dqn_weight[:, :min_features] = bc_weight[:, :min_features]
                        transferred_layers.append(f"{dqn_key} (partial)")
                    else:
                        skipped_layers.append((dqn_key, bc_state_dict[bc_key].shape, dqn_fe_state_dict[dqn_key].shape))
            else:
                skipped_layers.append((dqn_key, 'Not in BC model', dqn_fe_state_dict[dqn_key].shape))
        
        # Load updated weights
        feature_extractor.load_state_dict(dqn_fe_state_dict)
        
        print(f"\nTransferred layers ({len(transferred_layers)}):")
        for layer in transferred_layers:
            print(f"  - {layer}")
            
        print(f"\nSkipped layers ({len(skipped_layers)}):")
        for layer, bc_shape, dqn_shape in skipped_layers:
            print(f"  - {layer}: BC shape: {bc_shape}, DQN shape: {dqn_shape}")
            
        # Validation
        dummy_obs = dqn_model.env.reset()
        with th.no_grad():
            actions, _ = dqn_model.predict(dummy_obs, deterministic=True)
            print(f"Post-transfer prediction test successful - actions: {actions}")
                
    except Exception as e:
        print(f"Error during weight transfer: {str(e)}")
        raise

def make_env(rank):
    def _init():
        try:
            env = MinecraftEnv()
            return env
        except Exception as e:
            print(f"Error creating environment: {e}")
            raise
    return _init

class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None and self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            if self.best_model_save_path is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.best_model_save_path, 
                    f"best_model_{timestamp}_reward_{self.best_mean_reward:.2f}.zip"
                )
                self.model.save(save_path)
                print(f"New best model saved with reward {self.best_mean_reward:.2f}")
        return result

def create_new_model(env):
    debug_log("Starting model creation...")
    
    policy_kwargs = dict(
        features_extractor_class=BCMatchingFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 128]
    )
    
    try:
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
            tensorboard_log=LOG_DIR,
            device=device,
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        # Test model works by doing a prediction
        dummy_obs = env.reset()
        with th.no_grad():
            actions, _ = model.predict(dummy_obs, deterministic=True)
            debug_log(f"Initial prediction successful - actions: {actions}")
            
        return model
        
    except Exception as e:
        debug_log(f"Error creating model: {e}")
        debug_log(f"Policy kwargs: {policy_kwargs}")
        raise

def main():
    print("Select an option:")
    print("1. Load weights from BC model")
    print("2. Load existing DQN model")
    print("3. Start training from scratch")
    choice = input("Enter 1, 2, or 3: ").strip()

    # Create vectorized environment first
    env = SubprocVecEnv([make_env(i) for i in range(PARALLEL_ENVS)])
    env = VecMonitor(env)
    eval_env = SubprocVecEnv([make_env(0)])
    eval_env = VecMonitor(eval_env)

    print("Environment observation space:", env.observation_space)

    # Create initial model with better error handling
    try:
        model = create_new_model(env)
        print("Feature extractor initialized successfully:")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

    if choice == "1":
        # Load BC weights
        bc_models = [f for f in os.listdir(MODEL_PATH_BC) if f.endswith('.pth')]
        if not bc_models:
            print("No BC models found. Starting from scratch.")
        else:
            print("\nAvailable BC models:")
            for idx, name in enumerate(bc_models, 1):
                print(f"{idx}. {name}")
            model_idx = int(input("Select model number: ")) - 1
            bc_path = os.path.join(MODEL_PATH_BC, bc_models[model_idx])
            
            print("Transferring weights from BC model...")
            transfer_bc_weights_to_dqn(bc_path, model)

    elif choice == "2":
        # Load DQN weights
        dqn_models = [f for f in os.listdir(MODEL_PATH_DQN) if f.endswith('.zip')]
        if not dqn_models:
            print("No DQN models found. Starting from scratch.")
        else:
            print("\nAvailable DQN models:")
            for idx, name in enumerate(dqn_models, 1):
                print(f"{idx}. {name}")
            model_idx = int(input("Select model number: ")) - 1
            model = DQN.load(
                os.path.join(MODEL_PATH_DQN, dqn_models[model_idx]),
                env=env,
                device=device
            )

    # Setup callbacks
    callbacks = [
        SaveOnStepCallback(
            save_freq=SAVE_EVERY_STEPS,
            save_path=MODEL_PATH_DQN
        ),
        TimestampedEvalCallback(
            eval_env=eval_env,
            best_model_save_path=MODEL_PATH_DQN,
            log_path=LOG_DIR,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True
        )
    ]

    # Training loop with auto-restart
    timesteps_done = 0
    while timesteps_done < TOTAL_TIMESTEPS:
        try:
            remaining_timesteps = TOTAL_TIMESTEPS - timesteps_done
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                tb_log_name="dqn_from_bc",
                reset_num_timesteps=False
            )
            timesteps_done = TOTAL_TIMESTEPS  # Training completed successfully

        except Exception as e:
            print(f"Error during training: {e}")
            # Save recovery model
            recovery_path = os.path.join(
                MODEL_PATH_DQN, 
                f"recovery_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            model.save(recovery_path)
            print(f"Saved recovery model to {recovery_path}")
            
            # Update timesteps done
            timesteps_done = model.num_timesteps
            
            # Reset environments
            env.close()
            eval_env.close()
            
            # Recreate environments
            env = SubprocVecEnv([make_env(i) for i in range(PARALLEL_ENVS)])
            env = VecMonitor(env)
            eval_env = SubprocVecEnv([make_env(0)])
            eval_env = VecMonitor(eval_env)
            
            # Update model env reference
            model.set_env(env)
            
            print("Restarting training...")
            continue

    # Save final model
    final_path = os.path.join(MODEL_PATH_DQN, f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    model.save(final_path)
    print(f"Training completed. Final model saved to {final_path}")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
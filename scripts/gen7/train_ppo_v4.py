import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import datetime
import numpy as np
import torch as th
from typing import NamedTuple, Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import logging
from env_ppo import MinecraftEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import random
import traceback
from zipfile import ZipFile
import pygetwindow as gw
import fnmatch
import io
import time
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import safe_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau
import psutil
import torch
import gc
import torchvision.models as models


# Configuration
MODEL_PATH_PPO = r"E:\PPO_BC_MODELS\models_ppo_v4"
LOG_DIR_BASE = "tensorboard_logs_ppo_v4"
PARALLEL_ENVS = 8

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(LOG_DIR_BASE, f"run_{RUN_TIMESTAMP}")
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 3e-5  # PPO learning rate
N_STEPS = 1024
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
EVAL_FREQ = 4096
EVAL_EPISODES = 1
SAVE_EVERY_STEPS = 10000
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
USE_SDE = False
SDE_SAMPLE_FREQ = -1
TARGET_KL = 0.03
VERBOSE = 1
SEED = None

IMAGE_CNN_FREEZE = False
IMAGE_FUSION_FREEZE = False
INVENTORY_FREEZE = False
SURROUNDING_CNN_FREEZE = False
SURROUNDING_FUSION_FREEZE = False
FINAL_FUSION_FREEZE = False

IMAGE_CNN_GRAD_MULTIPLIER = 1.0
IMAGE_FUSION_GRAD_MULTIPLIER = 1.0
INVENTORY_GRAD_MULTIPLIER = 1.0
SURROUNDING_CNN_GRAD_MULTIPLIER = 1.0
SURROUNDING_FUSION_GRAD_MULTIPLIER = 1.0
FINAL_FUSION_GRAD_MULTIPLIER = 1.0

SPATIAL_WEIGHT = 0.8
CLASSIFICATION_WEIGHT = 0.2

SMALL_KERNEL_WEIGHT = 0.6  # Emphasize fine details
LARGE_KERNEL_WEIGHT = 0.4  # De-emphasize broader patterns

os.makedirs(MODEL_PATH_PPO, exist_ok=True)
os.makedirs(LOG_DIR_BASE, exist_ok=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
DEBUG_MODE = False

def debug_log(msg):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

def find_minecraft_windows():
    print("\n=== Starting Minecraft Window Detection ===")
    patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
    
    all_titles = gw.getAllTitles()
    matched_windows = []
    seen_handles = set()

    for title in all_titles:
        for pattern in patterns:
            if fnmatch.fnmatch(title, pattern):
                windows = gw.getWindowsWithTitle(title)
                for window in windows:
                    if window._hWnd not in seen_handles:
                        original_bounds = {
                            "left": window.left,
                            "top": window.top,
                            "width": window.width,
                            "height": window.height,
                        }

                        # Instead of using offsets, we now take a 224x224 area from the center
                        # Find the center of the window
                        center_x = original_bounds["left"] + original_bounds["width"] // 2
                        center_y = original_bounds["top"] + original_bounds["height"] // 2
                        crop_size = 224
                        half = crop_size // 2

                        adjusted_left = center_x - half
                        adjusted_top = center_y - half
                        adjusted_width = crop_size
                        adjusted_height = crop_size
                        
                        if adjusted_width <= 0 or adjusted_height <= 0:
                            print(f"Adjusted window size is non-positive for window: {title}. Skipping.")
                            continue
                        
                        adjusted_bounds = {
                            "left": adjusted_left,
                            "top": adjusted_top,
                            "width": adjusted_width,
                            "height": adjusted_height,
                        }
                        
                        matched_windows.append(adjusted_bounds)
                        seen_handles.add(window._hWnd)

    print("\nUnsorted adjusted window positions:")
    for i, window in enumerate(matched_windows):
        print(f"  Window {i+1}: ({window['left']}, {window['top']}) - {window['width']}x{window['height']}")

    potential_eval_windows = [w for w in matched_windows if w["left"] < 180 and w["top"] < 180]
    if not potential_eval_windows:
        raise ValueError("No suitable eval window found (needs both x < 100 and y < 100)")
    
    eval_window = min(potential_eval_windows, key=lambda w: (w["left"], w["top"]))
    matched_windows.remove(eval_window)

    print("\nEval window selected:")
    print(f"  ({eval_window['left']}, {eval_window['top']}) - {eval_window['width']}x{eval_window['height']}")
    small_x = [w for w in matched_windows if w["left"] < 180]
    large_x = [w for w in matched_windows if w["left"] >= 180]

    print("\nSmall x windows (unsorted):")
    for w in small_x:
        print(f"  ({w['left']}, {w['top']}) - {w['width']}x{w['height']}")

    print("\nLarge x windows (unsorted):")
    for w in large_x:
        print(f"  ({w['left']}, {w['top']}) - {w['width']}x{w['height']}")

    small_x_sorted = sorted(small_x, key=lambda w: w["top"])
    large_x_sorted = sorted(large_x, key=lambda w: w["top"])

    sorted_windows = [eval_window] + small_x_sorted + large_x_sorted

    print("\nFinal sorted windows:")
    print(f"Eval window: ({sorted_windows[0]['left']}, {sorted_windows[0]['top']})")
    print("Training windows:")
    for i, window in enumerate(sorted_windows[1:], 1):
        print(f"  Env {i}: ({window['left']}, {window['top']}) - {window['width']}x{window['height']}")

    return sorted_windows

def make_env(rank, is_eval=False, minecraft_windows=None):
    def _init():
        try:
            if is_eval:
                uri = "ws://localhost:8080"
                window_bounds = minecraft_windows[0]
            else:
                uri = f"ws://localhost:{8081 + rank}"
                window_bounds = minecraft_windows[rank + 1]

            window_title = "Minecraft"
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                raise ValueError(f"No window found with title '{window_title}' for rank {rank}.")

            if is_eval:
                window = windows[0]
            else:
                window = windows[rank + 1]

            if window.isMinimized:
                window.restore()
                time.sleep(0.1)

            env = MinecraftEnv(uri=uri, window_bounds=window_bounds)
            return env
        except Exception as e:
            print(f"Error creating {'evaluation' if is_eval else 'training'} environment {rank}: {e}")
            raise
    return _init

class TensorboardRewardLogger(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(TensorboardRewardLogger, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                self.episode_rewards.append(episode_reward)
                if DEBUG_MODE and self.writer is not None:
                    self.writer.add_scalar("Episode Reward", episode_reward, self.num_timesteps)
                if self.verbose > 0:
                    print(f"Logged episode reward: {episode_reward}")
        return True

    def _on_training_end(self):
        self.writer.close()


class GradNormCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super(GradNormCallback, self).__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        return True


class LRMonitorCallback(BaseCallback):
    def __init__(self, optimizer, verbose=1):
        super(LRMonitorCallback, self).__init__(verbose)
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
        )
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        return True

    def update_scheduler(self, value: float):
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(value)
        new_lr = self.optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            #print(f"LR changed from {old_lr} to {new_lr}")
            # Manually update the learning rate in the PPO algorithm
            self.model.learning_rate = new_lr
            # Update all optimizer groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr


class CustomEvalCallback(EvalCallback):
    def __init__(self, lr_monitor_callback: LRMonitorCallback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_monitor_callback = lr_monitor_callback

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            #rint(f"Passing reward {self.last_mean_reward:.4f} to LR scheduler")
            self.lr_monitor_callback.update_scheduler(self.last_mean_reward)
            # Log current learning rate
            current_lr = self.model.learning_rate
            #print(f"Current learning rate: {current_lr}")
        return result


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


def get_latest_model(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not models:
        return None
    latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)


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


################################################################################
#                       REPLACED FEATURE EXTRACTOR WITH RESNET18               #
################################################################################



# Configuration for weighting

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, debug=False, log_dir=None, get_step_fn=None):
        super().__init__(observation_space, features_dim)
        self.debug = debug
        self.writer = SummaryWriter(log_dir=log_dir) if (debug and log_dir is not None) else None
        self.get_step_fn = get_step_fn if get_step_fn is not None else (lambda: 0)

        self.img_head = nn.Sequential(
            # First block: subtle feature extraction (112x112 -> 56x56)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(2),  # 112x112 -> 56x56
            
            # Second block (56x56)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Add parallel conv paths with LeakyReLU
        self.final_small = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.final_large = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.final_combine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )


        # Calculate the flattened size (B, 32, 28, 28) -> B, 25088
        flattened_size = (64 + 2) * 56 * 56  # 66 channels now


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Additional networks for other observation components
        blocks_dim = observation_space.spaces["blocks"].shape[0]
        hand_dim = observation_space.spaces["hand"].shape[0]
        target_block_dim = observation_space.spaces["target_block"].shape[0]
        player_state_dim = observation_space.spaces["player_state"].shape[0]
        
        self.inventory_net = nn.Sequential(
            nn.Linear(hand_dim + blocks_dim + player_state_dim, 128),
            nn.ReLU(inplace=True),
        )

        surrounding_blocks_shape = observation_space.spaces["surrounding_blocks"].shape
        
        # Separate pathways for surrounding blocks and direction
        self.surrounding_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(surrounding_blocks_shape[0] * surrounding_blocks_shape[1] * surrounding_blocks_shape[2], 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.direction_processor = nn.Sequential(
            nn.Linear(2, 32),  # yaw and pitch
            nn.ReLU(inplace=True),
        )

        # Combined fusion layer for surrounding + direction
        self.surrounding_fusion = nn.Sequential(
            nn.Linear(96 + 32, 128),  # Combine processed features
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + 128 + 128, features_dim),
            nn.ReLU(inplace=True),
        )

        # Apply Freezing/Gradient scaling if needed
        self.freeze_layers()
        self.apply_gradient_scaling()

    def freeze_layers(self):
        if IMAGE_CNN_FREEZE:
            for param in self.img_head.parameters():
                param.requires_grad = False
            print("[INFO] Image CNN layers have been frozen.")

        if INVENTORY_FREEZE:
            for param in self.inventory_net.parameters():
                param.requires_grad = False
            print("[INFO] Inventory layers have been frozen.")

        if SURROUNDING_FUSION_FREEZE:
            for param in self.surrounding_fc.parameters():
                param.requires_grad = False
            print("[INFO] Surrounding Fusion layers have been frozen.")

        if FINAL_FUSION_FREEZE:
            for param in self.fusion_layer.parameters():
                param.requires_grad = False
            print("[INFO] Final Fusion layers have been frozen.")

    def apply_gradient_scaling(self):
        if any([
            IMAGE_CNN_GRAD_MULTIPLIER != 1.0,
            INVENTORY_GRAD_MULTIPLIER != 1.0,
            SURROUNDING_FUSION_GRAD_MULTIPLIER != 1.0,
            FINAL_FUSION_GRAD_MULTIPLIER != 1.0
        ]):
            self.register_hooks()

    def register_hooks(self):
        def get_multiplier(layer_name):
            return {
                'img_head': IMAGE_CNN_GRAD_MULTIPLIER,
                'inventory_net': INVENTORY_GRAD_MULTIPLIER,
                'surrounding_fc': SURROUNDING_FUSION_GRAD_MULTIPLIER,
                'fusion_layer': FINAL_FUSION_GRAD_MULTIPLIER
            }.get(layer_name, 1.0)

        layers = {
            'img_head': self.img_head,
            'inventory_net': self.inventory_net,
            'surrounding_fc': self.surrounding_fc,
            'fusion_layer': self.fusion_layer
        }

        for name, layer in layers.items():
            if layer is not None:
                multiplier = get_multiplier(name)
                if multiplier != 1.0:
                    for param in layer.parameters():
                        param.register_hook(lambda grad, m=multiplier: grad * m)
                    print(f"[INFO] Applied gradient multiplier {multiplier} to layer '{name}'.")

    def forward(self, observations):
        image = observations["image"]  
        blocks = observations["blocks"]
        hand = observations["hand"]
        target_spatial = observations["target_block"]
        player_state = observations["player_state"]
        surrounding_blocks = observations["surrounding_blocks"]

        # Extract image features
        x = self.img_head(image)
        small_features = self.final_small(x)
        large_features = self.final_large(x)
        
        # Combine image features
        combined = torch.cat([small_features, large_features], dim=1)
        img_feat = self.final_combine(combined)  # [B, 64, H, W]
        
        
        # Concatenate target block info with image features
        img_feat = torch.cat([img_feat, target_spatial], dim=1)  # Now 66 channels
        
        # Now apply FC layers to flattened features
        img_feat = self.fc(img_feat)

        # Extract inventory features
        inventory_input = th.cat([hand, blocks, player_state], dim=1)
        inv_feat = self.inventory_net(inventory_input)

        # Extract direction-aware surrounding features
        direction_feat = self.direction_processor(player_state[:, 3:5])  # yaw and pitch
        surrounding_feat = self.surrounding_processor(surrounding_blocks.flatten(1))
        surround_with_dir = torch.cat([surrounding_feat, direction_feat], dim=1)
        surround_feat = self.surrounding_fusion(surround_with_dir)

        # Final fusion
        fused = th.cat([img_feat, inv_feat, surround_feat], dim=1)
        final_features = self.fusion_layer(fused)

        return final_features
    



    def close(self):
        if self.writer is not None:
            self.writer.close()


def load_ppo_model(model_path, env, device):
    try:
        print(f"Loading PPO model weights from {model_path}...")
        
        # Create new model with current parameters
        new_model = create_new_model(env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        
        # Load old model
        old_model = PPO.load(model_path, env=env, device=device)
        
        # Transfer only the policy parameters
        new_model.policy.load_state_dict(old_model.policy.state_dict())
        
        print("Model weights transferred successfully.")
        print("Using current training parameters with loaded weights.")
        
        return new_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_ppo_model_partial(model_path, env, device):
    try:
        print(f"Loading PPO model weights from {model_path}...")
        
        # Create new model with current parameters
        new_model = create_new_model(env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        
        # Load old model
        old_model = PPO.load(model_path, env=env, device=device)
        
        # Get state dictionaries
        new_state_dict = new_model.policy.state_dict()
        old_state_dict = old_model.policy.state_dict()
        
        matched_state_dict = {
            k: v for k, v in old_state_dict.items()
            if k in new_state_dict and v.size() == new_state_dict[k].size()
        }
        
        unmatched_keys = [
            k for k in old_state_dict
            if k not in matched_state_dict
        ]
        
        new_state_dict.update(matched_state_dict)
        new_model.policy.load_state_dict(new_state_dict)
        
        print(f"Loaded {len(matched_state_dict)} matching layers out of {len(old_state_dict)} from the selected model.")
        if unmatched_keys:
            print(f"Could not load {len(unmatched_keys)} layers due to mismatches:")
            for key in unmatched_keys:
                print(f"  - {key}")
        else:
            print("All layers loaded successfully.")
        
        print("Using current training parameters with loaded weights.")
        
        return new_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def create_new_model(env, debug=True, log_dir=None):
    debug_log("Starting PPO model creation...")
    dummy_get_step_fn = lambda: 0

    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            debug=debug,
            log_dir=LOG_DIR,
            get_step_fn=dummy_get_step_fn
        ),
        net_arch=[256, 256]
    )

    # Create a constant learning rate function
    def constant_lr(_):
        return LEARNING_RATE

    model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        learning_rate=constant_lr,  # Use constant function instead of float
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        clip_range_vf=CLIP_RANGE_VF,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        use_sde=USE_SDE,
        sde_sample_freq=SDE_SAMPLE_FREQ,
        target_kl=TARGET_KL,
        tensorboard_log=LOG_DIR,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=VERBOSE,
        seed=SEED
    )

    # Force features_extractor creation
    dummy_obs = env.observation_space.sample()
    dummy_tensor = model.policy.obs_to_tensor(dummy_obs)[0]
    with th.no_grad():
        model.policy(dummy_tensor)

    return model

def log_memory_usage():
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    print(f"CPU Memory: {cpu_mem:.2f}MB, GPU Memory: {gpu_mem:.2f}MB")

def main():
    print("Select an option:")
    print("1. Load weights from a previous PPO model")
    print("2. Load existing PPO model")
    print("3. Start training from scratch")
    choice = input("Enter 1, 2, or 3: ").strip()

    minecraft_windows = find_minecraft_windows()
    
    train_env_fns = [make_env(i, minecraft_windows=minecraft_windows) for i in range(PARALLEL_ENVS)]
    train_env = SubprocVecEnv(train_env_fns)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    eval_env_fn = [make_env(0, is_eval=True, minecraft_windows=minecraft_windows)]
    eval_env = SubprocVecEnv(eval_env_fn)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False)

    print("Training Environment observation space:", train_env.observation_space)
    print("Evaluation Environment observation space:", eval_env.observation_space)

    if choice == "1":
        ppo_models = [f for f in os.listdir(MODEL_PATH_PPO) if f.endswith('.zip')]
        if not ppo_models:
            print("No PPO models found. Starting from scratch.")
            model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        else:
            print("\nAvailable PPO models:")
            for idx, name in enumerate(ppo_models, 1):
                print(f"{idx}. {name}")
            try:
                model_idx = int(input("Select the model number to transfer weights from: ")) - 1
                if model_idx < 0 or model_idx >= len(ppo_models):
                    raise IndexError("Selected model number is out of range.")
                selected_model = ppo_models[model_idx]
                model_path = os.path.join(MODEL_PATH_PPO, selected_model)
                
                print(f"Attempting to load weights from {selected_model}...")
                model = load_ppo_model_partial(model_path, train_env, device)
                
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                return
            except IndexError as ie:
                print(f"Error: {ie}")
                return
            except Exception as e:
                print(f"An unexpected error occurred during weight transfer: {e}")
                return
            

    elif choice == "2":
        ppo_models = [f for f in os.listdir(MODEL_PATH_PPO) if f.endswith('.zip')]
        if not ppo_models:
            print("No PPO models found. Starting from scratch.")
            model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        else:
            print("\nAvailable PPO models:")
            for idx, name in enumerate(ppo_models, 1):
                print(f"{idx}. {name}")
            try:
                model_idx = int(input("Select the model number to load: ")) - 1
                if model_idx < 0 or model_idx >= len(ppo_models):
                    raise IndexError("Selected model number is out of range.")
                selected_model = ppo_models[model_idx]
                model_path = os.path.join(MODEL_PATH_PPO, selected_model)
                print(f"Loading the selected PPO model from {model_path}...")
                model = load_ppo_model(model_path, train_env, device)
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                return
            except IndexError as ie:
                print(f"Error: {ie}")
                return
            except Exception as e:
                print(f"An unexpected error occurred during model loading: {e}")
                return

    else:
        model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        print("Training from scratch...")

    writer = SummaryWriter(LOG_DIR) if DEBUG_MODE else None

    lr_monitor_callback = LRMonitorCallback(
        optimizer=model.policy.optimizer,
        verbose=VERBOSE
    )

    eval_callback = CustomEvalCallback(
        lr_monitor_callback=lr_monitor_callback,
        eval_env=eval_env,
        best_model_save_path=MODEL_PATH_PPO,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False
    )
    grad_norm_callback = GradNormCallback(writer=writer)

    callbacks = [
        SaveOnStepCallback(
            save_freq=SAVE_EVERY_STEPS,
            save_path=MODEL_PATH_PPO
        ),
        eval_callback,
        grad_norm_callback,
        lr_monitor_callback  # Ensure LRMonitorCallback is always included
    ]

    timesteps_done = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    while timesteps_done < TOTAL_TIMESTEPS:
        try:
            if timesteps_done % 10000 == 0:
                log_memory_usage()
            debug_log(f"Starting PPO training iteration at timestep {timesteps_done}")
            remaining_timesteps = TOTAL_TIMESTEPS - timesteps_done
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                tb_log_name=f"ppo_from_bc_{RUN_TIMESTAMP}",
                reset_num_timesteps=False
            )
            timesteps_done = TOTAL_TIMESTEPS
            if timesteps_done % 10000 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during training: {e}")
            debug_log(f"Stack trace: {traceback.format_exc()}")
            recovery_path = os.path.join(
                MODEL_PATH_PPO, 
                f"recovery_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            model.save(recovery_path)
            print(f"Saved recovery model to {recovery_path}")
            
            latest_model_path = get_latest_model(MODEL_PATH_PPO)
            if (latest_model_path and latest_model_path != recovery_path):
                print(f"Loading the latest model from {latest_model_path} to continue training...")
                try:
                    model = load_ppo_model(latest_model_path, train_env, device)
                except Exception as load_e:
                    print(f"Failed to load the latest model: {load_e}")
                    break
            else:
                print("No previous models to load. Exiting training loop.")
                break
            
            train_env.close()
            eval_env.close()
            
            train_env_fns = [make_env(i, minecraft_windows=minecraft_windows) for i in range(PARALLEL_ENVS)]
            train_env = SubprocVecEnv(train_env_fns)
            train_env = VecMonitor(train_env)
            
            eval_env_fn = [make_env(0, is_eval=True, minecraft_windows=minecraft_windows)]
            eval_env = SubprocVecEnv(eval_env_fn)
            eval_env = VecMonitor(eval_env)
            
            model.set_env(train_env)
            
            print("Restarting training...")
            continue

    final_path = os.path.join(MODEL_PATH_PPO, f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    model.save(final_path)
    print(f"Training completed. Final PPO model saved to {final_path}")
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

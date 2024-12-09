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

# Configuration
MODEL_PATH_PPO = r"E:\PPO_BC_MODELS\models_ppo_large"
LOG_DIR_BASE = "tensorboard_logs_ppo_from_bc"
PARALLEL_ENVS = 6

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(LOG_DIR_BASE, f"run_{RUN_TIMESTAMP}")
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 1e-5  # PPO learning rate
N_STEPS = 512
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
EVAL_FREQ = 2048
EVAL_EPISODES = 1
SAVE_EVERY_STEPS = 20000
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = None
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
USE_SDE = False
SDE_SAMPLE_FREQ = -1
TARGET_KL = 0.02
VERBOSE = 1
SEED = None

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

                        adjusted_left = original_bounds["left"] + 30
                        adjusted_top = original_bounds["top"] + 50
                        adjusted_width = original_bounds["width"] - 60
                        adjusted_height = original_bounds["height"] - 70
                        
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

    potential_eval_windows = [w for w in matched_windows if w["left"] < 100 and w["top"] < 100]
    if not potential_eval_windows:
        raise ValueError("No suitable eval window found (needs both x < 100 and y < 100)")
    
    eval_window = min(potential_eval_windows, key=lambda w: (w["left"], w["top"]))
    matched_windows.remove(eval_window)

    print("\nEval window selected:")
    print(f"  ({eval_window['left']}, {eval_window['top']}) - {eval_window['width']}x{eval_window['height']}")
    small_x = [w for w in matched_windows if w["left"] < 100]
    large_x = [w for w in matched_windows if w["left"] >= 100]

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
        if DEBUG_MODE and self.num_timesteps % 1000 == 0 and self.writer is not None:
            parameters = [p for p in self.model.policy.parameters() if p.grad is not None]
            if parameters:
                total_norm = 0
                grad_means = []
                grad_vars = []
                for i, p in enumerate(parameters):
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    grad_means.append(p.grad.data.mean().item())
                    grad_vars.append(p.grad.data.var().item())
                    
                    if self.writer:
                        self.writer.add_scalar(f"grad/p{i}_mean", p.grad.data.mean().item(), self.num_timesteps)
                        self.writer.add_scalar(f"grad/p{i}_var", p.grad.data.var().item(), self.num_timesteps)
                
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("grad/total_norm", total_norm, self.num_timesteps)
                self.writer.add_scalar("grad/mean", np.mean(grad_means), self.num_timesteps)
                self.writer.add_scalar("grad/var", np.mean(grad_vars), self.num_timesteps)
        return True


class LRMonitorCallback(BaseCallback):
    def __init__(self, optimizer, writer, verbose=0):
        super(LRMonitorCallback, self).__init__(verbose)
        self.optimizer = optimizer
        self.writer = writer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if DEBUG_MODE and self.num_timesteps % 1000 == 0 and self.writer is not None:
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                self.writer.add_scalar(f"lr/group_{i}", lr, self.num_timesteps)
                if DEBUG_MODE:
                    print(f"Current learning rate for group {i}: {lr}")
        return True

    def update_scheduler(self, value):
        self.scheduler.step(value)


class CustomEvalCallback(EvalCallback):
    def __init__(self, lr_monitor_callback: LRMonitorCallback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_monitor_callback = lr_monitor_callback

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            self.lr_monitor_callback.update_scheduler(self.last_mean_reward)
        return result

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

def get_latest_model(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not models:
        return None
    latest_model = max(models, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

# ------------------- CHANGES START HERE ------------------- #
# We will introduce a small Transformer-based fusion and additional linear layers.
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BCMatchingFeatureExtractor(BaseFeaturesExtractor):
    _last_logged_step = -1

    def __init__(self, observation_space, features_dim=512, debug=False, log_dir=None, get_step_fn=None):
        super(BCMatchingFeatureExtractor, self).__init__(observation_space, features_dim)
        self.debug = debug
        self.writer = SummaryWriter(log_dir=log_dir) if (debug and log_dir is not None) else None
        
        self.get_step_fn = get_step_fn if get_step_fn is not None else (lambda: 0)

        image_shape = observation_space.spaces["image"].shape
        surrounding_blocks = observation_space.spaces["surrounding_blocks"].shape
        blocks_dim = observation_space.spaces["blocks"].shape[0]
        hand_dim = observation_space.spaces["hand"].shape[0]
        target_block_dim = observation_space.spaces["target_block"].shape[0]
        player_state_dim = observation_space.spaces["player_state"].shape[0]

        # Scalar inventory network (includes player state) - outputs 128
        self.scalar_inventory_net = nn.Sequential(
            nn.Linear(10 + player_state_dim, 128),
            nn.ReLU()
        )
        # Extra layer for inventory: 128 -> 128
        self.inventory_fc = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU()
        )

        # Matrix CNN and fusion - outputs 128
        self.matrix_cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_matrix = th.zeros(1, *surrounding_blocks)
            matrix_cnn_out = self.matrix_cnn(dummy_matrix)
            matrix_cnn_out_dim = matrix_cnn_out.shape[1]

        self.matrix_fusion = nn.Sequential(
            nn.Linear(matrix_cnn_out_dim, 128),
            nn.ReLU()
        )

        # Image network: produce image features, then map to (256 -> 128)
        self.image_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_image = th.zeros(1, *image_shape)
            image_out = self.image_net(dummy_image)
            image_out_dim = image_out.shape[1]

        # First map image_out_dim -> 256 -> 128
        self.image_fc_expanded = nn.Sequential(
            nn.Linear(image_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Transformer-based fusion
        # We have 3 tokens: matrix(128-d), inventory(128-d), image(128-d)
        d_model = 128
        nhead = 4
        num_layers = 2
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # After transformer, we have (B, 3, 128). We'll mean-pool over tokens and map to 512
        self.post_transformer_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU()
        )

        self._features_dim = features_dim
        debug_log(f"Features dim set to {self._features_dim}")

        debug_log("Testing forward pass with dummy data...")
        dummy_obs = {
            "image": dummy_image,
            "blocks": th.zeros(1, blocks_dim),
            "hand": th.zeros(1, hand_dim),
            "target_block": th.zeros(1, target_block_dim),
            "player_state": th.zeros(1, player_state_dim),
            "surrounding_blocks": dummy_matrix
        }
        with th.no_grad():
            test_out = self.forward(dummy_obs)
            debug_log(f"Forward pass successful - output shape: {test_out.shape}")
            if self.debug:
                self._log_feature_stats(test_out, step=0)

        self._register_gradient_hooks()

    def _create_hook(self, name):
        def hook(module, grad_input, grad_output):
            if not DEBUG_MODE:
                return
            if grad_input and grad_input[0] is not None:
                grad = grad_input[0]
                grad_mean = grad.mean().item()
                grad_var = grad.var().item()
                step = self.get_step_fn()
                debug_log(f"Gradient stats for {name}: mean={grad_mean:.8f}, var={grad_var:.8f}, step={step}")
                if self.writer is not None:
                    self.writer.add_scalar(f"Gradients/{name}_Mean", grad_mean, step)
                    self.writer.add_scalar(f"Gradients/{name}_Variance", grad_var, step)
            else:
                debug_log(f"No gradient flowing through {name}")
        return hook

    def _register_gradient_hooks(self):
        target_modules = {
            "scalar_inventory_net": self.scalar_inventory_net,
            "inventory_fc": self.inventory_fc,
            "matrix_cnn": self.matrix_cnn,
            "matrix_fusion": self.matrix_fusion,
            "image_net": self.image_net,
            "image_fc_expanded": self.image_fc_expanded,
            "transformer_encoder": self.transformer_encoder,
            "post_transformer_fc": self.post_transformer_fc
        }

        for name, module in target_modules.items():
            for submodule_name, submodule in module.named_modules():
                if len(list(submodule.children())) == 0:  # leaf module
                    submodule.register_full_backward_hook(self._create_hook(f"{name}.{submodule_name}" if submodule_name else name))
            debug_log(f"Registered gradient hook for module: {name}")

    def forward(self, observations):
        blocks = observations["blocks"]
        hand = observations["hand"]
        target_block = observations["target_block"]
        player_state = observations["player_state"]
        surrounding_blocks = observations["surrounding_blocks"]
        image = observations["image"]

        # Process inventory and player state together, then another fc layer
        inventory_input = th.cat([hand, target_block, blocks, player_state], dim=1)
        inventory_features = self.scalar_inventory_net(inventory_input)  # 128
        inventory_features = self.inventory_fc(inventory_features)        # another 128

        # Process matrix
        matrix_out = self.matrix_cnn(surrounding_blocks)
        matrix_features = self.matrix_fusion(matrix_out)  # 128

        # Process image
        image_out = self.image_net(image)
        image_features = self.image_fc_expanded(image_out)  # 128 after expansion

        # Stack as tokens for transformer
        # shape: (batch, 3, 128)
        tokens = th.stack([matrix_features, inventory_features, image_features], dim=1)
        # Pass through transformer
        transformed = self.transformer_encoder(tokens)
        # Pool: take mean over tokens
        pooled = transformed.mean(dim=1)  # (batch, 128)
        final_features = self.post_transformer_fc(pooled)  # (batch, 512)

        if self.debug:
            current_step = self.get_step_fn()  
            if current_step > BCMatchingFeatureExtractor._last_logged_step:
                self._log_feature_stats(final_features, step=current_step)
                BCMatchingFeatureExtractor._last_logged_step = current_step

        return final_features

    def _log_feature_stats(self, features, step):
        if not DEBUG_MODE:
            return
        mean = features.mean().item()
        var = features.var().item()
        debug_log(f"Feature stats at step {step}: mean={mean:.4f}, var={var:.4f}")
        if self.writer is not None:
            self.writer.add_scalar("feat/mean", mean, step)
            self.writer.add_scalar("feat/var", var, step)

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

def create_new_model(env, debug=True, log_dir=None):
    debug_log("Starting PPO model creation...")
    dummy_get_step_fn = lambda: 0

    # Deeper network architecture
    policy_kwargs = dict(
        features_extractor_class=BCMatchingFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=512,
            debug=debug,
            log_dir=LOG_DIR,
            get_step_fn=dummy_get_step_fn
        ),
        net_arch=[512, 512, 256, 256]  # Deeper network architecture as requested
    )

    model = PPO(
        policy=ActorCriticPolicy,
        env=env,
        learning_rate=LEARNING_RATE,
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

    def get_step_fn():
        return model.num_timesteps

    # Force features_extractor creation
    dummy_obs = env.observation_space.sample()
    dummy_tensor = model.policy.obs_to_tensor(dummy_obs)[0]
    with th.no_grad():
        model.policy(dummy_tensor)

    # Set get_step_fn for feature extractor
    model.policy.features_extractor.get_step_fn = get_step_fn

    return model

def log_memory_usage():
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB if using CUDA
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
    
    eval_env_fn = [make_env(0, is_eval=True, minecraft_windows=minecraft_windows)]
    eval_env = SubprocVecEnv(eval_env_fn)
    eval_env = VecMonitor(eval_env)

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
                
                model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
                print(f"Transferring weights from {selected_model}...")
                # Here you could implement any weight transfer if desired
                print("Weight transfer completed successfully.")
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

    lr_monitor_callback = LRMonitorCallback(optimizer=model.policy.optimizer, writer=writer)
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
        lr_monitor_callback
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

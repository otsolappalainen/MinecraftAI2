import os
import datetime
import numpy as np
import torch as th
from typing import NamedTuple, Dict, Any, Optional
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
import logging
from env_dqn_paraller import MinecraftEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import create_mlp
import random
import traceback
from zipfile import ZipFile
import pygetwindow as gw
import fnmatch
import io
import time
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import DictReplayBuffer, DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import safe_mean
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
MODEL_PATH_BC = "models_bc"
MODEL_PATH_DQN = r"E:\DQN_BC_MODELS\models_dqn_from_bc"
LOG_DIR_BASE = "tensorboard_logs_dqn_from_bc"
PARALLEL_ENVS = 6

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(LOG_DIR_BASE, f"run_{RUN_TIMESTAMP}")
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 2e-4
BUFFER_SIZE = 10000
BATCH_SIZE = 256
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1500
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.05
EVAL_FREQ = 1000
EVAL_EPISODES = 2
SAVE_EVERY_STEPS = 5000

os.makedirs(MODEL_PATH_DQN, exist_ok=True)
os.makedirs(LOG_DIR_BASE, exist_ok=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")

DEBUG_MODE = True  # Toggle for verbose logging

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
                for p in parameters:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    grad_means.append(p.grad.data.mean().item())
                    grad_vars.append(p.grad.data.var().item())
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Gradients/TotalNorm", total_norm, self.num_timesteps)
                self.writer.add_scalar("Gradients/MeanGrad", np.mean(grad_means), self.num_timesteps)
                self.writer.add_scalar("Gradients/VarGrad", np.mean(grad_vars), self.num_timesteps)
        return True

class LRMonitorCallback(BaseCallback):
    def __init__(self, optimizer, writer, verbose=0):
        super(LRMonitorCallback, self).__init__(verbose)
        self.optimizer = optimizer
        self.writer = writer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10, verbose=DEBUG_MODE)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if DEBUG_MODE and self.num_timesteps % 1000 == 0 and self.writer is not None:
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group['lr']
                self.writer.add_scalar(f"LearningRate/group_{i}", lr, self.num_timesteps)
        return True

    def update_scheduler(self, mean_reward):
        self.scheduler.step(mean_reward)

class CustomEvalCallback(EvalCallback):
    def __init__(self, lr_monitor_callback: LRMonitorCallback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_monitor_callback = lr_monitor_callback

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            self.lr_monitor_callback.update_scheduler(self.last_mean_reward)
        return result

class MultiTokenAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, tokens):
        # tokens: [Batch, 3, Dim]
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = attended.mean(dim=1) # [Batch, Dim]
        return attended

class BCMatchingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, debug=False, log_dir=None, get_step_fn=None):
        super(BCMatchingFeatureExtractor, self).__init__(observation_space, features_dim)
        self.debug = debug
        self.writer = SummaryWriter(log_dir=log_dir) if (debug and log_dir is not None) else None
        
        self._current_step = 0
        self._initialized = False
        self.debug = debug
        self.writer = SummaryWriter(log_dir=log_dir) if (debug and log_dir is not None) else None
        self.get_step_fn = get_step_fn if get_step_fn is not None else (lambda: 0)

        image_shape = observation_space.spaces["image"].shape
        surrounding_blocks = observation_space.spaces["surrounding_blocks"].shape
        blocks_dim = observation_space.spaces["blocks"].shape[0]
        hand_dim = observation_space.spaces["hand"].shape[0]
        target_block_dim = observation_space.spaces["target_block"].shape[0]
        player_state_dim = observation_space.spaces["player_state"].shape[0]

        self.scalar_inventory_net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU()
        )

        self.player_state_net = nn.Sequential(
            nn.Linear(player_state_dim, 32),
            nn.ReLU()
        )

        self.matrix_cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_matrix = th.zeros(1, *surrounding_blocks)
            matrix_cnn_out = self.matrix_cnn(dummy_matrix)
            matrix_cnn_out_dim = matrix_cnn_out.shape[1]

        self.matrix_fusion = nn.Sequential(
            nn.Linear(matrix_cnn_out_dim, 64),
            nn.ReLU()
        )

        self.spatial_fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU()
        )

        self.image_net = nn.Sequential(
            nn.Conv2d(image_shape[0], 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_image = th.zeros(1, *image_shape)
            image_out = self.image_net(dummy_image)
            image_out_dim = image_out.shape[1]

        self.image_fc = nn.Sequential(
            nn.Linear(image_out_dim, 64),
            nn.ReLU()
        )

        class ResidualBlock(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(features, features),
                    nn.ReLU(),
                    nn.Linear(features, features)
                )

            def forward(self, x):
                return F.relu(x + self.block(x))

        # Use attention to fuse tokens
        self.token_attention = MultiTokenAttention(dim=64, num_heads=4)

        self.fusion_layer = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Linear(256, features_dim),
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
            test_out = self.forward(dummy_obs, test_mode=True)
            debug_log(f"Forward pass successful - output shape: {test_out.shape}")
            if self.debug:
                self._log_feature_stats(test_out, step=0)

        self._num_timesteps = 0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @num_timesteps.setter
    def num_timesteps(self, steps):
        self._num_timesteps = steps

    @property 
    def current_step(self):
        return self._current_step
        
    @current_step.setter
    def current_step(self, step):
        self._current_step = step
        self._initialized = True

    def forward(self, observations, test_mode=False):
        if not self._initialized and not test_mode:
            debug_log("Warning: Feature extractor not properly initialized with step count")
            
        blocks = observations["blocks"]
        hand = observations["hand"]
        target_block = observations["target_block"]
        player_state = observations["player_state"]
        surrounding_blocks = observations["surrounding_blocks"]
        image = observations["image"]

        inventory_input = th.cat([hand, target_block, blocks], dim=1)
        inventory_features = self.scalar_inventory_net(inventory_input) * 1.2

        player_state_features = self.player_state_net(player_state)

        matrix_out = self.matrix_cnn(surrounding_blocks)
        matrix_features = self.matrix_fusion(matrix_out)

        spatial_input = th.cat([matrix_features, player_state_features], dim=1)
        spatial_features = self.spatial_fusion(spatial_input)

        image_out = self.image_net(image)
        image_features = self.image_fc(image_out)

        tokens = th.stack([spatial_features, inventory_features, image_features], dim=1) # [B,3,64]
        attended = self.token_attention(tokens) # [B,64]

        final_features = self.fusion_layer(attended) # [B,256]

        if self.debug:
            current_step = self.get_step_fn()
            self._log_feature_stats(final_features, step=current_step)

        return final_features

    def _log_feature_stats(self, features, step):
        if not DEBUG_MODE:
            return
        mean = features.mean().item()
        var = features.var().item()
        debug_log(f"Feature stats at step {step}: mean={mean:.4f}, var={var:.4f}")
        if self.writer is not None:
            self.writer.add_scalar("Features/Mean", mean, step)
            self.writer.add_scalar("Features/Variance", var, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()

def transfer_bc_weights_to_dqn(bc_model_path, dqn_model):
    try:
        print(f"Loading BC model from {bc_model_path}")
        bc_state_dict = th.load(bc_model_path, map_location=device, weights_only=True)
        
        feature_extractor = BCMatchingFeatureExtractor(dqn_model.observation_space, debug=DEBUG_MODE, log_dir=LOG_DIR)
        feature_extractor = feature_extractor.to(device)
        dqn_model.policy.features_extractor = feature_extractor
        dqn_fe_state_dict = feature_extractor.state_dict()
        
        transferred_layers = []
        skipped_layers = []
        
        for dqn_key in dqn_fe_state_dict.keys():
            bc_key = dqn_key
            if bc_key in bc_state_dict:
                if bc_state_dict[bc_key].shape == dqn_fe_state_dict[dqn_key].shape:
                    dqn_fe_state_dict[dqn_key].copy_(bc_state_dict[bc_key])
                    transferred_layers.append(dqn_key)
                else:
                    skipped_layers.append((dqn_key, bc_state_dict[bc_key].shape, dqn_fe_state_dict[dqn_key].shape))
            else:
                skipped_layers.append((dqn_key, 'Not in BC model', dqn_fe_state_dict[dqn_key].shape))
        
        feature_extractor.load_state_dict(dqn_fe_state_dict)
        
        print(f"\nTransferred layers ({len(transferred_layers)}):")
        for layer in transferred_layers:
            print(f"  - {layer}")
            
        print(f"\nSkipped layers ({len(skipped_layers)}):")
        for layer, bc_shape, dqn_shape in skipped_layers:
            print(f"  - {layer}: BC shape: {bc_shape}, DQN shape: {dqn_shape}")
            
        dummy_obs = dqn_model.env.reset()
        with th.no_grad():
            actions, _ = dqn_model.predict(dummy_obs, deterministic=True)
            print(f"Post-transfer prediction test successful - actions: {actions}")
                
    except Exception as e:
        print(f"Error during weight transfer: {str(e)}")
        raise

def create_new_model(env, debug=True, log_dir=None):
    debug_log("Starting model creation...")
    dummy_get_step_fn = lambda: 0
    
    policy_kwargs = dict(
        features_extractor_class=BCMatchingFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            debug=debug,
            log_dir=LOG_DIR,
            get_step_fn=dummy_get_step_fn
        ),
        net_arch=[256, 256]
    )

    try:
        model = DQN(
            policy=DQNPolicy,
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

        def get_step_fn():
            return model.num_timesteps
        model.policy.features_extractor.get_step_fn = get_step_fn

        return model
    except Exception as e:
        debug_log(f"Error creating model: {e}")
        debug_log(f"Policy kwargs: {policy_kwargs}")
        raise

def transfer_weights(old_model, new_model):
    old_state_dict = old_model.policy.state_dict()
    new_state_dict = new_model.policy.state_dict()

    transferred_layers = []
    skipped_layers = []

    for key, param in old_state_dict.items():
        if key in new_state_dict:
            if new_state_dict[key].shape == param.shape:
                new_state_dict[key] = param
                transferred_layers.append(key)
            else:
                skipped_layers.append((key, param.shape, new_state_dict[key].shape))
        else:
            skipped_layers.append((key, param.shape, 'Not present in new model'))

    new_model.policy.load_state_dict(new_state_dict, strict=False)
    print("Weight transfer completed.")
    print(f"Transferred layers ({len(transferred_layers)}):")
    for layer in transferred_layers:
        print(f"  - {layer}")
    print(f"Skipped layers ({len(skipped_layers)}):")
    for layer, old_shape, new_shape in skipped_layers:
        print(f"  - {layer}: {old_shape} -> {new_shape}")

def read_dqn_model_metadata(model_path):
    try:
        with ZipFile(model_path, 'r') as zip_ref:
            print("\nZIP file contents:")
            for info in zip_ref.filelist:
                print(f"  {info.filename}: {info.file_size} bytes")
            
            if 'data' in zip_ref.namelist():
                with zip_ref.open('data') as f:
                    content = f.read()
                    print("\nModel metadata:", content[:1000], "...")
                    
            if 'policy.pth' in zip_ref.namelist():
                print("\nPolicy file found")
    except Exception as e:
        print(f"Error reading model file: {e}")
        raise

def raw_transfer_dqn_weights(model_path, new_model):
    print(f"\nTransferring weights from: {model_path}")
    
    try:
        new_state_dict = new_model.policy.state_dict()
        print("\nNew model layers:")
        for key, tensor in new_state_dict.items():
            print(f"  {key}: {tensor.shape}")

        transferred = []
        skipped = []

        with ZipFile(model_path, 'r') as zip_ref:
            print("\nZIP contents:")
            for info in zip_ref.filelist:
                print(f"  {info.filename}: {info.file_size} bytes")

            if 'policy.pth' in zip_ref.namelist():
                with zip_ref.open('policy.pth') as f:
                    buffer = io.BytesIO(f.read())
                    old_state_dict = th.load(buffer, map_location=device)
                    
                    print("\nLoaded model layers:")
                    for key, tensor in old_state_dict.items():
                        print(f"  {key}: {tensor.shape}")

                    for new_key in new_state_dict.keys():
                        if new_key in old_state_dict:
                            old_tensor = old_state_dict[new_key]
                            new_tensor = new_state_dict[new_key]
                            
                            if old_tensor.shape == new_tensor.shape:
                                new_state_dict[new_key].copy_(old_tensor)
                                transferred.append(new_key)
                            elif new_key.startswith("features_extractor.scalar_inventory_net") and new_key in old_state_dict:
                                min_dim = min(old_tensor.shape[1], new_tensor.shape[1])
                                new_state_dict[new_key][:, :min_dim].copy_(old_tensor[:, :min_dim])
                                transferred.append(f"{new_key} (partial)")
                            else:
                                skipped.append((new_key, old_tensor.shape, new_tensor.shape))
                        else:
                            skipped.append((new_key, "Not found in old model", new_state_dict[new_key].shape))

        new_model.policy.load_state_dict(new_state_dict, strict=False)

        print(f"\nTransferred {len(transferred)} layers:")
        for layer in transferred:
            print(f"  + {layer}")

        print(f"\nSkipped {len(skipped)} layers:")
        for layer, old_shape, new_shape in skipped:
            print(f"  - {layer}: old={old_shape}, new={new_shape}")

        return new_model

    except Exception as e:
        print(f"Error during transfer: {e}")
        print(traceback.format_exc())
        raise

def load_dqn_model(model_path, env, device):
    try:
        print(f"Loading DQN model from {model_path}...")
        model = DQN.load(model_path, env=env, device=device)
        print("Model loaded successfully.")

        # Update learning parameters
        model.learning_rate = LEARNING_RATE
        model.buffer_size = BUFFER_SIZE
        model.batch_size = BATCH_SIZE
        model.gamma = GAMMA
        model.train_freq = TRAIN_FREQ
        model.gradient_steps = GRADIENT_STEPS
        model.target_update_interval = TARGET_UPDATE_INTERVAL
        model.exploration_fraction = EXPLORATION_FRACTION
        model.exploration_final_eps = EXPLORATION_FINAL_EPS

        print("Updated model with current learning parameters.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

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

def main():
    print("Select an option:")
    print("1. Load weights from a previous DQN model")
    print("2. Load existing DQN model")
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
        dqn_models = [f for f in os.listdir(MODEL_PATH_DQN) if f.endswith('.zip')]
        if not dqn_models:
            print("No DQN models found. Starting from scratch.")
            model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        else:
            print("\nAvailable DQN models:")
            for idx, name in enumerate(dqn_models, 1):
                print(f"{idx}. {name}")
            try:
                model_idx = int(input("Select the model number to transfer weights from: ")) - 1
                if model_idx < 0 or model_idx >= len(dqn_models):
                    raise IndexError("Selected model number is out of range.")
                selected_model = dqn_models[model_idx]
                model_path = os.path.join(MODEL_PATH_DQN, selected_model)
                
                model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
                print(f"Transferring weights from {selected_model}...")
                raw_transfer_dqn_weights(model_path, model)
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
        dqn_models = [f for f in os.listdir(MODEL_PATH_DQN) if f.endswith('.zip')]
        if not dqn_models:
            print("No DQN models found. Starting from scratch.")
            model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        else:
            latest_model_path = get_latest_model(MODEL_PATH_DQN)
            if latest_model_path:
                print(f"Loading the latest DQN model from {latest_model_path}...")
                model = load_dqn_model(latest_model_path, train_env, device)
            else:
                print("No models found. Starting from scratch.")
                model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)

    else:
        model = create_new_model(train_env, debug=DEBUG_MODE, log_dir=LOG_DIR)
        print("Training from scratch...")

    writer = SummaryWriter(LOG_DIR) if DEBUG_MODE else None

    lr_monitor_callback = LRMonitorCallback(optimizer=model.policy.optimizer, writer=writer)
    eval_callback = CustomEvalCallback(
        lr_monitor_callback=lr_monitor_callback,
        eval_env=eval_env,
        best_model_save_path=MODEL_PATH_DQN,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False
    )
    grad_norm_callback = GradNormCallback(writer=writer)

    # Ensure FeatureExtractorStepCallback runs first
    callbacks = [
        SaveOnStepCallback(
            save_freq=SAVE_EVERY_STEPS,
            save_path=MODEL_PATH_DQN
        ),
        eval_callback,
        grad_norm_callback,
        lr_monitor_callback
    ]

    timesteps_done = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    while timesteps_done < TOTAL_TIMESTEPS:
        try:
            debug_log(f"Starting training iteration at timestep {timesteps_done}")
            remaining_timesteps = TOTAL_TIMESTEPS - timesteps_done
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                tb_log_name=f"dqn_from_bc_{RUN_TIMESTAMP}",
                reset_num_timesteps=False
            )
            timesteps_done = TOTAL_TIMESTEPS

        except Exception as e:
            print(f"Error during training: {e}")
            debug_log(f"Stack trace: {traceback.format_exc()}")
            recovery_path = os.path.join(
                MODEL_PATH_DQN, 
                f"recovery_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            )
            model.save(recovery_path)
            print(f"Saved recovery model to {recovery_path}")
            
            latest_model_path = get_latest_model(MODEL_PATH_DQN)
            if (latest_model_path and latest_model_path != recovery_path):
                print(f"Loading the latest model from {latest_model_path} to continue training...")
                try:
                    model = load_dqn_model(latest_model_path, train_env, device)
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

    final_path = os.path.join(MODEL_PATH_DQN, f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    model.save(final_path)
    print(f"Training completed. Final model saved to {final_path}")
    
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

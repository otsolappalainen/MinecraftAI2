import os
import numpy as np
import torch as th
import torch.nn as nn
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import logging
from torch.utils.data import DataLoader, Dataset
from imitation.data.types import transitions_collate_fn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define ACTION_MAPPING
ACTION_MAPPING = {
    0: "move_forward",
    1: "move_backward",
    2: "move_left",
    3: "move_right",
    4: "jump_walk_forward",
    5: "jump",
    6: "sneak",
    7: "look_left",
    8: "look_right",
    9: "look_up",
    10: "look_down",
    11: "turn_left",
    12: "turn_right",
    13: "attack",
    14: "use",
    15: "next_item",
    16: "previous_item",
    17: "no_op"
}

# Define observation and action spaces
observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3 * 224 * 224 + 8,), dtype=np.float32)
action_space = gym.spaces.Discrete(len(ACTION_MAPPING))

# Combine observations into a single tensor
def combine_obs(image, other):
    image_flat = image.view(image.size(0), -1)  # Flatten image to [N, 3*224*224]
    return th.cat([image_flat, other], dim=1)

# Device setup (GPU or CPU)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Generate placeholder data
num_samples = 2046
obs = {
    'image': th.tensor(np.random.rand(num_samples, 3, 224, 224).astype(np.float32), device=device),
    'other': th.tensor(np.random.rand(num_samples, 8).astype(np.float32), device=device),
}
actions = th.tensor(np.random.randint(0, len(ACTION_MAPPING), size=(num_samples,), dtype=np.int64), device=device)
next_obs = {
    'image': th.tensor(np.random.rand(num_samples, 3, 224, 224).astype(np.float32), device=device),
    'other': th.tensor(np.random.rand(num_samples, 8).astype(np.float32), device=device),
}
dones = th.tensor(np.zeros(num_samples, dtype=bool), device=device)
infos = [{} for _ in range(num_samples)]

# Combine observations
obs = combine_obs(obs['image'], obs['other'])
next_obs = combine_obs(next_obs['image'], next_obs['other'])

# Log detailed information
logging.info(f"obs has shape {obs.shape}")
logging.info(f"actions has shape {actions.shape}")
logging.info(f"next_obs has shape {next_obs.shape}")
logging.info(f"dones has {len(dones)} timesteps")

# Transitions Dataset
class TransitionsDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions.obs)

    def __getitem__(self, idx):
        return {
            "obs": self.transitions.obs[idx],
            "acts": self.transitions.acts[idx],
            "next_obs": self.transitions.next_obs[idx],
            "dones": self.transitions.dones[idx],
            "infos": self.transitions.infos[idx],
        }

# DataLoader Collate Function
def safe_collate_fn(batch):
    """
    Ensures all tensors in the batch are moved to CPU for compatibility with NumPy.
    """
    for sample in batch:
        for key, value in sample.items():
            if isinstance(value, th.Tensor) and value.device.type == "cuda":
                sample[key] = value.cpu()  # Move tensors to CPU
    return transitions_collate_fn(batch)

# Create Transitions object
transitions = Transitions(
    obs=obs,
    acts=actions.cpu(),  # Ensure actions are on CPU
    next_obs=next_obs,
    dones=dones.cpu().numpy(),  # Convert to NumPy boolean array
    infos=infos
)

# Wrap transitions in a dataset and DataLoader
transitions_dataset = TransitionsDataset(transitions)
data_loader = DataLoader(
    dataset=transitions_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=safe_collate_fn,
    pin_memory=True if th.cuda.is_available() else False,
)

# Initialize BC trainer
bc_trainer = BC(
    observation_space=observation_space,
    action_space=action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(0),
    device=device,
)

# Replace BC trainer's DataLoader
bc_trainer.data_loader = data_loader

# Train the policy
bc_trainer.train(n_epochs=10)

logging.info("Behavioral Cloning training completed.")

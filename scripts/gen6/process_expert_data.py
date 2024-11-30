import pickle
import numpy as np
from imitation.data.types import Transitions

# Load the expert data
with open(r'C:\Users\odezz\source\MinecraftAI2\scripts\gen6\expert_data\session_20241130_012423\expert_data.pkl', 'rb') as f:
    expert_data = pickle.load(f)

# Prepare data for Transitions
observations = []
actions = []
next_observations = []
dones = []
infos = []

for entry in expert_data:
    observations.append(entry['observation'])
    actions.append(entry['action'])
    next_observations.append(entry['next_observation'])
    dones.append(entry['done'])
    infos.append(entry['info'])

# Convert lists to arrays
# Flatten the observation dictionaries
obs_image = np.array([obs['image'] for obs in observations], dtype=np.float32)  # Shape: (2046, 3, 224, 224)
obs_other = np.array([obs['other'] for obs in observations], dtype=np.float32)  # Shape: (2046, 8)

next_obs_image = np.array([obs['image'] for obs in next_observations], dtype=np.float32)  # Shape: (2046, 3, 224, 224)
next_obs_other = np.array([obs['other'] for obs in next_observations], dtype=np.float32)  # Shape: (2046, 8)

# Combine the image and other components into a single flat array
obs_flat = np.hstack((obs_image.reshape(len(obs_image), -1), obs_other))  # Shape: (2046, 150560)
next_obs_flat = np.hstack((next_obs_image.reshape(len(next_obs_image), -1), next_obs_other))  # Shape: (2046, 150560)

# Convert actions and other components
actions = np.array(actions, dtype=np.int64)  # Shape: (2046,)
dones = np.array(dones, dtype=bool)  # Shape: (2046,)

# Create Transitions object
transitions = Transitions(
    obs=obs_flat,         # Flattened observations
    acts=actions,         # Actions
    next_obs=next_obs_flat,  # Flattened next observations
    dones=dones,          # Done flags
    infos=infos           # Info dictionary
)

print("Transitions object created successfully!")

import numpy as np
import torch as th
from imitation.algorithms.bc import BC
from imitation.util.logger import configure
from imitation.data.types import Transitions
import gymnasium as gym

# Synthetic Data
obs_flat = np.random.randn(100, 150556).astype(np.float32)
actions = np.random.randint(0, 10, size=100).astype(np.int64)
next_obs_flat = obs_flat.copy()
dones = np.zeros(100, dtype=bool)
infos = [{} for _ in range(100)]

transitions = Transitions(
    obs=obs_flat,
    acts=actions,
    next_obs=next_obs_flat,
    dones=dones,
    infos=infos,
)

# Observation and Action Spaces
obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_flat.shape[1],), dtype=np.float32)
action_space = gym.spaces.Discrete(np.max(actions) + 1)

# Initialize BC Trainer
bc_trainer = BC(
    observation_space=obs_space,
    action_space=action_space,
    demonstrations=transitions,
    batch_size=32,
    optimizer_cls=th.optim.Adam,
    optimizer_kwargs={"lr": 1e-4},
    rng=np.random.default_rng(seed=42),
    custom_logger=configure("test_bc_logs", ["stdout"]),
    device="cpu",
)

# Train
bc_trainer.train(n_epochs=1)
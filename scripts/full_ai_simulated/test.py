
from cnn_env import SimulatedEnvGraphics  # Import the environment

env = SimulatedEnvGraphics()
print(env.observation_space)
print(env.observation_space["image"].dtype)  # Should be np.float32
print(env.observation_space["other"].dtype)
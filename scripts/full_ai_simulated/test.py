
from cnn_env import SimulatedEnvGraphics  # Import the environment

env = SimulatedEnvGraphics()
print(env.observation_space)
print(env.observation_space["image"].dtype)  # Should be np.float32
print(env.observation_space["other"].dtype)


obs, _ = env.reset()
print("Observation from reset:", obs)
print("Image type:", type(obs["image"]))
print("Other type:", type(obs["other"]))
print("Image shape:", obs["image"].shape)
print("Other shape:", obs["other"].shape)
print("Image dtype:", obs["image"].dtype)
print("Other dtype:", obs["other"].dtype)

# Step through the environment
obs, reward, done, truncated, info = env.step(env.action_space.sample())
print("Observation from step:", obs)
print("Image type:", type(obs["image"]))
print("Other type:", type(obs["other"]))
print("Image shape:", obs["image"].shape)
print("Other shape:", obs["other"].shape)
print("Image dtype:", obs["image"].dtype)
print("Other dtype:", obs["other"].dtype)
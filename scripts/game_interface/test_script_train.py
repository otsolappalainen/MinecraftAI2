from minecraft_env import MinecraftEnv
import numpy as np

# Initialize the environment
env = MinecraftEnv()

# Reset the environment and print the initial observation
obs = env.reset()
print(f"Initial Observation: {obs}")

# Run a few steps in the environment manually to see if `done` triggers immediately
done = False
step_count = 0
while not done and step_count < 20:  # Limiting to 20 steps to avoid an infinite loop if there's an issue
    # Sample a random action
    action = env.action_space.sample()  # Random action to test environment logic
    obs, reward, done, info = env.step(action)
    
    # Print out details of each step
    print(f"Step: {step_count}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
    
    # Check for an immediate `done` flag or unexpected end of episode
    if done:
        print("Episode ended early.")
    
    step_count += 1

# Close the environment after testing
env.close()
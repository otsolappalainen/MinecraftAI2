import numpy as np
import random
import time
from env import MinecraftEnv  # Make sure this is the correct import path


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"



def test_env_initialization():
    print(f"{GREEN}Testing environment initialization...{RESET}")
    env = MinecraftEnv()
    assert isinstance(env.action_space, type(env.action_space)), "Action space not defined correctly"
    assert isinstance(env.observation_space, type(env.observation_space)), "Observation space not defined correctly"
    print("Environment initialized successfully.")
    return env

def test_reset_function(env):
    print(f"{GREEN}Testing reset function...{RESET}")
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), "Reset observation is not an ndarray"
    assert obs.shape == (6,), f"Observation shape mismatch, expected (6,) but got {obs.shape}"
    print("Reset function works as expected.")
    return obs

def test_step_function(env):
    print(f"{GREEN}Testing step function...{RESET}")
    
    # Start by resetting the environment
    env.reset()
    
    for action in range(env.action_space.n):
        print(f"\n{GREEN}Testing action {action} ({env.action_map[action]})...{RESET}")
        
        # Perform a step
        obs, reward, done, truncated, _ = env.step(action)
        
        # Check the shape and type of the observation
        assert isinstance(obs, np.ndarray), "Step observation is not an ndarray"
        assert obs.shape == (6,), f"Observation shape mismatch, expected (6,) but got {obs.shape}"
        
        # Check that reward is a float (it should be positive, negative, or zero)
        assert isinstance(reward, (int, float)), f"Reward type mismatch, expected int or float but got {type(reward)}"
        
        # Check the done and truncated flags
        assert isinstance(done, bool), "Done flag is not a boolean"
        assert isinstance(truncated, bool), "Truncated flag is not a boolean"
        
        print(f"Action {env.action_map[action]} | Observation: {obs} | Reward: {reward} | Done: {done} | Truncated: {truncated}")
        
        # Reset if episode ended
        if done or truncated:
            env.reset()

def test_agent_interaction(env):
    print("Testing agent-side interactions...")
    env.reset()
    actions = list(env.action_map.keys())
    
    for _ in range(5):
        action = random.choice(actions)
        obs, reward, done, truncated, _ = env.step(action)
        print(f"Agent action: {env.action_map[action]} | Observation: {obs} | Reward: {reward}")
        
        # If done or truncated, reset environment for further testing
        if done or truncated:
            env.reset()

def test_model_interaction(env):
    print("Testing AI model-style interaction (random actions)...")
    env.reset()
    
    for step in range(10):
        action = env.action_space.sample()  # Random action for testing model interaction
        obs, reward, done, truncated, _ = env.step(action)
        print(f"Model step {step + 1} | Action: {env.action_map[action]} | Observation: {obs} | Reward: {reward}")
        
        # Reset if the episode ended
        if done or truncated:
            print("Episode finished. Resetting environment.")
            env.reset()

if __name__ == "__main__":
    # Initialize and test environment
    env = test_env_initialization()

    # Test the reset functionality
    test_reset_function(env)

    # Test the step functionality for each action
    test_step_function(env)

    # Test interaction from the agent's side
    test_agent_interaction(env)

    # Test interaction from a random AI model's side
    test_model_interaction(env)

    # Close the environment after testing
    env.close()

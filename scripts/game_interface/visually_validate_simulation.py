import os
import time
from stable_baselines3 import PPO
from env import SimpleMinecraftEnv  # Import your training environment

def load_model_and_simulate():
    # Initialize the environment with rendering enabled
    env = SimpleMinecraftEnv(enable_rendering=True)
    model_path = r"C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\logs\best_model.zip"  # Update the path as needed

    # Check if the model file exists
    if os.path.exists(model_path):
        print("Loading model...")
        model = PPO.load(model_path, env=env)
        obs, _ = env.reset()  # Reset the environment to get initial observation
        done = False
        print("Starting simulation...")

        try:
            while not done:
                # Predict the action using the trained model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take a step in the environment using the predicted action
                obs, reward, done, truncated, info = env.step(action)
                x, z, yaw = obs
                # Render the environment (handles pygame events inside)
                env.render()

                # Print debugging information
                print(f"Action taken: {action}, X: {x}, Z: {z}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

                # Control the simulation speed
                time.sleep(0.1)
        except Exception as e:
            print("An error occurred during simulation:", e)
        finally:
            env.close()
    else:
        print("Model file not found!")

if __name__ == '__main__':
    load_model_and_simulate()

import os
import time
import threading
from pynput import keyboard
from stable_baselines3 import PPO
from minecraft_env import MinecraftEnv  # Import your real Minecraft environment
import numpy as np


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# Path to the folder containing trained models
model_folder_path = r"C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\logs"

# Initialize the real Minecraft environment
env = MinecraftEnv()
print("Environment initialized.")

# Event to control the start of simulation
start_simulation_event = threading.Event()

# Function to monitor for the "I" key press
def monitor_start_key():
    def on_press(key):
        if key == keyboard.KeyCode.from_char('i'):
            print("Start key 'I' pressed. Beginning simulation...")
            start_simulation_event.set()
            return False  # Stop the listener

    print("Press 'I' to start the simulation.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Start the keyboard monitoring in a separate thread
keyboard_thread = threading.Thread(target=monitor_start_key)
keyboard_thread.daemon = True  # Daemonize to close with the main program
keyboard_thread.start()

# Function to prompt the user to select a model
def select_model():
    # List all .zip files in the model folder
    model_files = [f for f in os.listdir(model_folder_path) if f.endswith('.zip')]
    if not model_files:
        print("No model files found in the specified directory.")
        return None

    # Display available models
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i + 1}: {model_file}")

    # Prompt user to select a model
    choice = int(input("Select a model by entering the corresponding number: ")) - 1
    if 0 <= choice < len(model_files):
        selected_model = os.path.join(model_folder_path, model_files[choice])
        print(f"Selected model: {selected_model}")
        return selected_model
    else:
        print("Invalid selection.")
        return None

# Load the model and run the simulation once the "I" key is pressed
def load_model_and_simulate():
    # Prompt user to select a model
    model_path = select_model()
    if model_path is None:
        return  # Exit if no valid model was selected

    # Check if the model file exists
    if os.path.exists(model_path):
        print("Loading model...")
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")
        
        # Wait for the "I" key press to start simulation
        start_simulation_event.wait()

        obs, _ = env.reset()  # Reset the environment and get the initial observation
        done = False
        print("Starting simulation in real environment without graphics...")

        if len(obs) != 6:
            print(f"Unexpected initial observation length: {len(obs)}. Observation: {obs}")
            obs = np.zeros(6, dtype=np.float32)


        try:
            while not done:
                if not np.all(np.isfinite(obs)):
                    print("Non-finite values detected in observation. Resetting to default values.")
                    obs = np.zeros(6, dtype=np.float32)

                move_start_time = time.time()
                # Predict the action using the trained model
                action, _ = model.predict(obs, deterministic=True)

                # Take a step in the environment using the predicted action
                obs, reward, done, truncated, info = env.step(action)
                move_duration = time.time() - move_start_time
                
                # Safely unpack the observation if it has 6 values
                if len(obs) == 6 and np.all(np.isfinite(obs)):
                    x, z, yaw, pitch, target_x, target_z = obs
                else:
                    print("Unexpected observation values. Skipping this step.")
                    obs = np.zeros(6, dtype=np.float32)  # Reset obs if it has invalid values
                    continue

                # Print debugging information
                print(f"Action {action} took {GREEN}{move_duration:.4f}{RESET} seconds to complete")
                print(f"Target X: {GREEN}{target_x}{RESET}, Target Z: {GREEN}{target_z}{RESET}, Done: {done}, Truncated: {truncated}")

                # Control the simulation speed (adjust delay as needed)
                time.sleep(0.01)
                elapsed_time = time.time() - move_start_time
                print(f"Total elapsed time: {GREEN}{elapsed_time:.2f}{RESET} seconds")

            print("Simulation completed.")
        except Exception as e:
            print("An error occurred during simulation:", e)
        finally:
            env.close()
    else:
        print("Model file not found!")

if __name__ == '__main__':
    load_model_and_simulate()


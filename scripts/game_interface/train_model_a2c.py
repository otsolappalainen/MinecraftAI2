import os
import threading
import argparse
from pynput.keyboard import Listener, KeyCode
from stable_baselines3 import PPO, A2C  # Import both PPO and A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

# Import custom environments
from minecraft_env import MinecraftEnv
from env import SimpleMinecraftEnv

# Declare global flags
global stop_training, start_training, model, model_path, best_model_base_name
stop_training = False
start_training = False

# Define default training parameters
training_params = {
    "n_steps": 4048,
    "batch_size": 256,
    "learning_rate": 0.0005,
    "n_epochs": 24,
    "total_timesteps": 10000
}

class PlottingCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=1, log_file='training_log.csv'):
        super(PlottingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.total_rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.num_episodes = 0
        self.log_file = log_file

        # Write header only once if the file is empty
        if not os.path.isfile(self.log_file) or os.stat(self.log_file).st_size == 0:
            self.init_csv()

    def init_csv(self):
        """Write the CSV header."""
        try:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Total Steps', 'Average Reward'])
        except Exception as e:
            print(f"Failed to initialize CSV file: {e}")

    def _on_step(self) -> bool:
        """Log rewards at each step."""
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        self.episode_rewards.extend(rewards)
        self.episode_lengths.extend([1] * len(rewards))

        for idx, done in enumerate(dones):
            if done:
                episode_reward = sum(self.episode_rewards)
                self.total_rewards.append(episode_reward)
                self.num_episodes += 1
                self.episode_rewards = []
                self.episode_lengths = []

                if self.num_episodes % self.log_freq == 0:
                    mean_reward = np.mean(self.total_rewards[-self.log_freq:])
                    total_steps = self.num_timesteps
                    self.write_to_csv(self.num_episodes, total_steps, mean_reward)
                    print(f"Episode {self.num_episodes}: Average Reward over last {self.log_freq} episodes: {mean_reward:.2f}")

        return True

    def write_to_csv(self, episode, total_steps, mean_reward):
        """Append training data to CSV file."""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_steps, mean_reward])
        except Exception as e:
            print(f"Failed to write to CSV: {e}")

def make_env(seed):
    def _init():
        env = SimpleMinecraftEnv()
        env.seed(seed)
        return env
    return _init

def select_model(directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"):
    model_files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if model_files:
        print("Available models:")
        for idx, model_file in enumerate(model_files, 1):
            print(f"{idx}: {model_file}")
        choice = int(input("Select a model to continue training (enter number): ")) - 1
        return os.path.join(directory, model_files[choice])
    else:
        print("No existing models found.")
        return None

def initialize_model(env, model_type="PPO", directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"):
    global model_path
    
    # Initialize model_path as None
    model_path = None

    # Ask user if they want to start from scratch or load an existing model
    start_fresh = input("Do you want to start with a fresh model? (yes/no): ").strip().lower() == "yes"

    # Choose model class based on model_type argument
    model_class = PPO if model_type == "PPO" else A2C  # Extend with other model types if needed
    
    if not start_fresh:
        # Try to load an existing model
        model_path = select_model(directory)
    
    if model_path and not start_fresh:
        print(f"Loading the selected model from {model_path}...")
        return model_class.load(model_path, env=env, verbose=1)
    else:
        print(f"Starting training from scratch with a new {model_type} model.")
        
        # Adjust arguments for A2C, which doesn't use `n_epochs` or `batch_size` as PPO does
        model_kwargs = {
            "env": env,
            "learning_rate": training_params["learning_rate"],
            "n_steps": training_params["n_steps"],
            "verbose": 1,
        }
        if model_type == "PPO":
            model_kwargs["batch_size"] = training_params["batch_size"]
            model_kwargs["n_epochs"] = training_params["n_epochs"]

        return model_class("MlpPolicy", **model_kwargs)

def monitor_keyboard():
    global stop_training, start_training, model_path
    def on_press(key):
        global stop_training, start_training
        if key == KeyCode.from_char('i'):
            if not start_training:
                print("Start key 'I' pressed. Beginning training...")
                start_training = True
        elif key == KeyCode.from_char('q'):
            if start_training:
                print("Exit key 'Q' pressed. Saving model and stopping training...")
                save_model_with_numbered_name()
                stop_training = True
                return False

    print("Press 'I' to start training or 'Q' to save and exit.")
    with Listener(on_press=on_press) as listener:
        listener.join()

def save_model_with_numbered_name():
    global model, best_model_base_name
    directory = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"
    index = 1
    while True:
        model_name = f"{best_model_base_name}_{index}.zip"
        model_path = os.path.join(directory, model_name)
        if not os.path.exists(model_path):
            model.save(model_path)
            print(f"Model saved as {model_name}")
            break
        index += 1

def plot_training_metrics(log_file='training_log.csv'):
    episodes = []
    total_steps = []
    average_rewards = []

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Episode'].isdigit():
                try:
                    episodes.append(int(row['Episode']))
                    total_steps.append(int(row['Total Steps']))
                    average_rewards.append(float(row['Average Reward']))
                except ValueError:
                    continue

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, average_rewards, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["PPO", "A2C"], default="PPO", help="Select the RL model to use (PPO or A2C)")
    parser.add_argument('--simulate', action='store_true', help="Use the simulated environment for training")
    parser.add_argument('--num_envs', type=int, default=1, help="Number of parallel environments (only for simulation)")
    args = parser.parse_args()

    if args.simulate:
        env = DummyVecEnv([make_env(seed) for seed in range(args.num_envs)])
        model_directory = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"
    else:
        env = MinecraftEnv()
        model_directory = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"

    # Prompt for best model base name
    best_model_base_name = input("Enter the base name for the best models: ")

    # Initialize model with the selected type (PPO or A2C)
    model = initialize_model(env, model_type=args.model)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/', name_prefix=best_model_base_name)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000, deterministic=True, render=False, callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=1000000, verbose=1))
    plotting_callback = PlottingCallback(log_freq=5000)

    keyboard_thread = threading.Thread(target=monitor_keyboard)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    try:
        while not start_training:
            if stop_training:
                break
        print("Starting the training loop...")
        iteration = 0
        while not stop_training:
            iteration_start = time.time()
            print(f"Training loop iteration: {iteration}")
            model.learn(total_timesteps=training_params["total_timesteps"], callback=[checkpoint_callback, eval_callback, plotting_callback])
            iteration_end = time.time()
            print(f"Iteration {iteration} completed in {iteration_end - iteration_start:.2f} seconds.")
            if stop_training:
                break
            iteration += 1
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        env.close()
        plot_training_metrics(log_file='training_log.csv')

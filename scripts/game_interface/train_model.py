import os
import threading
from datetime import datetime
import argparse
from pynput.keyboard import Listener, KeyCode
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import csv

# Import custom environments
from minecraft_env import MinecraftEnv
from env import SimpleMinecraftEnv

# Declare global flags
global stop_training, start_training, model, model_path
stop_training = False
start_training = False

# Define model name here so it can be accessed later
model_name = "ppo_minecraft_agent"

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

def unique_model_filename(base_name="ppo_agent"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.zip"

def initialize_model(env, directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"):
    global model_path  # Allow this to be accessible in other functions
    best_model_name = "best_model.zip"
    model_path = os.path.join(directory, best_model_name)
    
    # Normalize and print the path to verify
    model_path = os.path.normpath(model_path)
    print(f"Looking for model at: {model_path}")

    if os.path.isfile(model_path):
        choice = input(f"Model '{best_model_name}' found. Do you want to continue training it? (Y/N): ").strip().lower()
        if choice == 'y':
            print(f"Loading the existing model from {model_path}...")
            return PPO.load(model_path, env=env, verbose=1)
        elif choice == 'n':
            print("Starting training from scratch with a new model.")
            return PPO(
                "MlpPolicy",
                env,
                n_steps=2048,           # Number of steps to collect before each update
                batch_size=64,          # Size of each training
                learning_rate=0.0002,     # Learning rate for the optimizer
                n_epochs=10,            # Number of epochs for each update
                verbose=1
            )
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")
            return initialize_model(env, directory)  # Recursively prompt again
    else:
        print(f"No existing model found at {model_path}. Training a new model from scratch.")
        return PPO(
                "MlpPolicy",
                env,
                n_steps=2048,           # Number of steps to collect before each update
                batch_size=64,          # Size of each training
                learning_rate=0.0002,     # Learning rate for the optimizer
                n_epochs=10,            # Number of epochs for each update
                verbose=1
                )

def monitor_keyboard():
    global stop_training, start_training, model
    def on_press(key):
        global stop_training, start_training, model_path
        if key == KeyCode.from_char('i'):
            if not start_training:
                print("Start key 'I' pressed. Beginning training...")
                start_training = True
        elif key == KeyCode.from_char('q'):
            if start_training:
                print("Exit key 'Q' pressed. Saving model and stopping training...")
                model.save(model_path)  # Save to defined model_path
                stop_training = True
                return False

    print("Press 'I' to start training or 'Q' to save and exit.")
    with Listener(on_press=on_press) as listener:
        listener.join()

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
    parser.add_argument('--simulate', action='store_true', help="Use the simulated environment for training")
    parser.add_argument('--num_envs', type=int, default=1, help="Number of parallel environments (only for simulation)")
    args = parser.parse_args()

    if args.simulate:
        env = DummyVecEnv([make_env(seed) for seed in range(args.num_envs)])
        model_directory = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"
    else:
        env = MinecraftEnv()
        model_directory = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"

    model = initialize_model(env, model_directory)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/', name_prefix=model_name)
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
            print(f"Training loop iteration: {iteration}")
            model.learn(total_timesteps=5000, callback=[checkpoint_callback, eval_callback, plotting_callback])
            print("Finished model.learn() call.")
            if stop_training:
                break
            iteration += 1
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        env.close()
        plot_training_metrics(log_file='training_log.csv')







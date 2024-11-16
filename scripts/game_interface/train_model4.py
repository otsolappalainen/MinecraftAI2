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
import time

# Import custom environments
from minecraft_env import MinecraftEnv
from env import SimpleMinecraftEnv

# File path for the training log
LOG_FILE_PATH = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\training_log.csv"

# Function to clear the log file and write the header
def clear_log_file(log_file_path):
    """Clears the log file and writes the header."""
    with open(log_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total Steps', 'Average Reward'])
    print(f"Log file {log_file_path} cleared and header written.")

# Call clear_log_file once at the start of the script
clear_log_file(LOG_FILE_PATH)

# Declare global flags
global stop_training, start_training, model, model_path, best_model_base_name
stop_training = False
start_training = False

# Define default training parameters
training_params = {
    "n_steps": 4096,
    "batch_size": 128,
    "learning_rate": 0.0003,
    "n_epochs": 10,
    "total_timesteps": 200000
}

# Define difficulty settings
difficulty_settings = {
    "easy": {
        "arena_size": 25,
        "noise_level": 0.0,
        "target_distance": 10,
        "target_area_size": 2,
        "random_target_mode": False,
        "target_coordinates": (0, 0),
        "max_episode_length": 500
    },
    "medium": {
        "arena_size": 50,
        "noise_level": 0.05,
        "target_distance": 25,
        "target_area_size": 10,
        "random_target_mode": False,
        "target_coordinates": (10, 10),
        "max_episode_length": 1000
    },
    "hard": {
        "arena_size": 100,
        "noise_level": 0.1,
        "target_distance": 50,
        "target_area_size": 15,
        "random_target_mode": True,
        "max_episode_length": 2000
    }
}

# Custom EvalCallback with timestamped model saving
class TimestampedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(TimestampedEvalCallback, self).__init__(*args, **kwargs)
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        result = super(TimestampedEvalCallback, self)._on_step()
        if self.best_mean_reward < self.last_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            if self.best_model_save_path:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_model_path = os.path.join(self.best_model_save_path, f"best_model_{current_time}.zip")
                self.model.save(timestamped_model_path)
                print(f"New best model saved with timestamp: {timestamped_model_path}")
        return result

# PlottingCallback to log rewards and clear log file at the start
class PlottingCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=1, log_file=LOG_FILE_PATH):
        super(PlottingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.total_rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.num_episodes = 0
        self.log_file = log_file

        # Clear the log file and write header on initialization
        clear_log_file(self.log_file)

    def _on_step(self) -> bool:
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
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, total_steps, mean_reward])
        except Exception as e:
            print(f"Failed to write to CSV: {e}")

def make_env(seed, difficulty):
    def _init():
        env = SimpleMinecraftEnv(
            arena_size=difficulty_settings[difficulty]["arena_size"],
            noise_level=difficulty_settings[difficulty]["noise_level"],
            target_distance=difficulty_settings[difficulty]["target_distance"],
            target_area_size=difficulty_settings[difficulty]["target_area_size"],
            random_target_mode=difficulty_settings[difficulty].get("random_target_mode", True),
            target_coordinates=difficulty_settings[difficulty].get("target_coordinates", (0, 0)),
            max_episode_length=difficulty_settings[difficulty]["max_episode_length"]
        )
        env.seed(seed)
        return env
    return _init

def select_model(directory):
    model_files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    
    if model_files:
        print("Available models:")
        for idx, model_file in enumerate(model_files, 1):
            print(f"{idx}: {model_file}")
        model_choice = int(input("Select a model to continue training (enter number): ")) - 1
        return os.path.join(directory, model_files[model_choice])
    else:
        print("No existing models found in this difficulty level.")
        return None

def initialize_model(env, directory, difficulty):
    global model_path
    choice = input("Do you want to load an existing model or start fresh? (load/fresh): ").strip().lower()

    if choice == "load":
        model_path = select_model(directory)
        if model_path:
            print(f"Loading the selected model from {model_path}...")
            return PPO.load(model_path, env=env, verbose=1)
    else:
        print("Starting training from scratch with a new model.")
    
    # Start a new model if not loading
    return PPO(
        "MlpPolicy",
        env,
        device="cpu",
        n_steps=training_params["n_steps"],
        batch_size=training_params["batch_size"],
        learning_rate=training_params["learning_rate"],
        n_epochs=training_params["n_epochs"],
        ent_coef=0.01,
        verbose=1
    )

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

def save_model_with_numbered_name(difficulty):
    global model, best_model_base_name
    directory = os.path.join("C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs", difficulty)
    os.makedirs(directory, exist_ok=True)
    index = 1
    while True:
        model_name = f"{best_model_base_name}_{index}.zip"
        model_path = os.path.join(directory, model_name)
        if not os.path.exists(model_path):
            model.save(model_path)
            print(f"Model saved as {model_name} in {directory}")
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
    parser.add_argument('--simulate', action='store_true', help="Use the simulated environment for training")
    parser.add_argument('--num_envs', type=int, default=1, help="Number of parallel environments (only for simulation)")
    parser.add_argument('--threaded', action='store_true', help="Run the environment in a multi-process threaded mode")
    args = parser.parse_args()

    # Prompt user to select difficulty level
    difficulty = input("Select training difficulty (easy, medium, hard): ").strip().lower()
    if difficulty not in difficulty_settings:
        print("Invalid difficulty level. Defaulting to 'easy'.")
        difficulty = "easy"

    # Set up the environment based on `--threaded` argument
    env_fn = SubprocVecEnv if args.threaded else DummyVecEnv
    env = env_fn([make_env(seed, difficulty) for seed in range(args.num_envs)])
    model_directory = os.path.join("C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs", difficulty)

    # Prompt for best model base name
    best_model_base_name = input("Enter the base name for the best models: ")

    model = initialize_model(env, model_directory, difficulty)

    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=model_directory, name_prefix=best_model_base_name)
    plotting_callback = PlottingCallback(log_freq=200)
    eval_callback = TimestampedEvalCallback(
        env,
        best_model_save_path=model_directory,
        log_path=model_directory,
        eval_freq=1000,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=1000000, verbose=1)
    )

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

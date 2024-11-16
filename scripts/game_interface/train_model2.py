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
import optuna

# Import custom environments
from minecraft_env import MinecraftEnv
from env import SimpleMinecraftEnv

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

OPTUNA_LOG_FILE = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\optuna_trials_log.csv"




# Declare global flags
global stop_training, start_training, model, model_path, best_model_base_name
stop_training = False
start_training = False

# Define default training parameters
default_params = {
    "n_steps": 2048,
    "batch_size": 128,
    "learning_rate": 0.00015,
    "n_epochs": 10,
    "total_timesteps": 5000
}

def clear_log_files():
    """Clears specified log files at the beginning of an Optuna optimization run and writes headers."""
    for file_path in [OPTUNA_LOG_FILE]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleared log file: {file_path}")

            # Write headers after clearing the file or if the file did not exist
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Trial", "Value (Reward)", "n_steps", "batch_size", "learning_rate", "n_epochs"])
                print("Headers written to:", file_path)
                
        except Exception as e:
            print(f"Error clearing log file {file_path}: {e}")




class PlottingCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=1, log_file=OPTUNA_LOG_FILE):
        super(PlottingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.total_rewards = []
        self.num_episodes = 0
        self.log_file = log_file

    def _on_step(self) -> bool:
        """Log rewards at each step and log average reward periodically."""
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        
        # Track rewards for completed episodes only
        self.total_rewards.extend([sum(rewards) for done in dones if done])
        
        # Log the average reward periodically
        if len(self.total_rewards) >= self.log_freq:
            mean_reward = np.mean(self.total_rewards[-self.log_freq:])
            total_steps = self.num_timesteps
            self.write_to_csv(self.num_episodes, total_steps, mean_reward)
            print(f"Episode {self.num_episodes}: Average Reward over last {self.log_freq} episodes: {mean_reward:.2f}")
        
        return True

    def write_to_csv(self, episode, total_steps, mean_reward):
        """Append training data to the main CSV file."""
        try:
            with open(self.log_file, 'a', newline="") as f:
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


def select_model(directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs", optuna_mode=False):
    """
    Select a model from the directory to continue training or start a new model.
    If optuna_mode is True, it automatically starts a new model without asking.
    """
    if optuna_mode:
        print("Optuna mode detected. Starting a new model.")
        return None  # Automatically choose to start a new model

    model_files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if model_files:
        print("Available models:")
        for idx, model_file in enumerate(model_files, 1):
            print(f"{idx}: {model_file}")
        choice = int(input("Do you want to (1) continue training an existing model or (2) start a new model? (enter 1 or 2): "))
        if choice == 1:
            model_choice = int(input("Select a model to continue training (enter number): ")) - 1
            return os.path.join(directory, model_files[model_choice])
        else:
            print("Starting a new model.")
            return None
    else:
        print("No existing models found. Starting a new model.")
        return None

"""
n_steps = trial.suggest_int("n_steps", 2048, 4096, step=512)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])  # Use fixed options
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        n_epochs = trial.suggest_int("n_epochs", 20, 30)

"""




def initialize_model(env, trial, optuna_mode=False, directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"):
    global model_path
    model_path = select_model(directory, optuna_mode=optuna_mode)
    
    if model_path:
        print(f"Loading the selected model from {model_path}...")
        return PPO.load(model_path, env=env, verbose=1)
    else:
        # Define the hyperparameters to tune using the trial object
        n_steps = 8192
        batch_size = 256  # Use fixed options
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        n_epochs = 20

        print("Starting training from scratch with a new model.")
        return PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
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

def plot_training_metrics(log_file='optuna_trials_log.csv'):
    trials = []
    rewards = []

    # Read the file and populate the data lists
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ensure the required columns exist in the row before accessing
            if 'Trial' in row and 'Value (Reward)' in row:
                try:
                    trials.append(int(row['Trial']))
                    rewards.append(float(row['Value (Reward)']))
                except ValueError:
                    continue  # Skip any rows with invalid data

    # Plot the rewards over trials
    plt.figure(figsize=(10, 5))
    plt.plot(trials, rewards, marker='o', color='b', label='Reward')
    plt.xlabel("Trial")
    plt.ylabel("Reward")
    plt.title("Reward Over Trials")
    plt.grid(True)
    plt.legend()
    plt.show()

with open(OPTUNA_LOG_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Trial", "Value (Reward)", "n_steps", "batch_size", "learning_rate", "n_epochs"])


def objective(trial):
    best_model_base_name = "optuna_best_model"

    n_steps = 4096
    batch_size = 512  # Use fixed options
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    n_epochs = 24

    # Set up environment
    env = DummyVecEnv([make_env(seed) for seed in range(args.num_envs)]) if args.simulate else SimpleMinecraftEnv()

    # Set trial number for each individual environment in DummyVecEnv
    if args.simulate:
        for i in range(env.num_envs):
            if hasattr(env.envs[i], 'set_trial_number'):
                env.envs[i].set_trial_number(trial.number)  # Set trial number in each environment
    else:
        env.set_trial_number(trial.number)  # Directly set trial number if not using DummyVecEnv

    # Initialize the PPO model with the suggested parameters and fixed values
    model = initialize_model(env, trial, optuna_mode=args.optuna_mode)

    # Define callbacks
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/', name_prefix=best_model_base_name)
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000, deterministic=True, render=False, callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=1000000, verbose=1))
    plotting_callback = PlottingCallback(log_freq=1000)

    # Train the model with a fixed number of timesteps
    total_timesteps = 300000  # Set the desired training timesteps
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, plotting_callback])

    # Evaluate and return the mean reward as the objective value
    mean_reward = np.mean(eval_callback.best_mean_reward)
    env.close()  # Close the environment

    # Log trial results to CSV, using fixed values where applicable
    with open(OPTUNA_LOG_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([trial.number, mean_reward, n_steps, batch_size, learning_rate, n_epochs])

    return mean_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true', help="Use the simulated environment for training")
    parser.add_argument('--num_envs', type=int, default=1, help="Number of parallel environments (only for simulation)")
    parser.add_argument('--optuna_mode', action='store_true', help="Flag to enable Optuna hyperparameter tuning mode")
    args = parser.parse_args()

    if args.optuna_mode:
        clear_log_files()

    # Set up Optuna study and run optimization
    if args.optuna_mode:
        import optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)  # Adjust n_trials as needed
    else:
        pass

    # Use optuna_mode flag in select_model
    model_path = select_model(directory="C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs", optuna_mode=args.optuna_mode)


    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    plot_training_metrics(log_file=OPTUNA_LOG_FILE)

import os
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import datetime
import numpy as np
import optuna  # For hyperparameter optimization
from env_simulated import SimulatedEnvGraphics  # Import the environment

# Hardcoded paths and parameters
MODEL_PATH = r"E:\model_spam"  # Path to save the trained model
BEST_MODELS_PATH = os.path.join(MODEL_PATH, "best_models")
TRAJECTORY_PATH = os.path.join(MODEL_PATH, "trajectories")
PARALLEL_ENVS = 1  # Number of parallel environments
RENDER_MODE = "none"
SAVE_EVERY_STEPS = 100000  # Save the model every 100000 steps
OPTUNA_TRIALS = 40  # Number of trials for hyperparameter tuning

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback to save the model every N steps.
    """
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        timesteps = self.num_timesteps
        if timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_step_{timesteps}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved at step {timesteps} to {save_file}")
        return True


def make_env(render_mode="none"):
    """
    Create a simulated environment instance.
    """
    def _init():
        return SimulatedEnvGraphics(render_mode=render_mode)
    return _init


class SaveBestModelOnEvalCallback(BaseCallback):
    """
    Custom callback to save the best model during evaluation with console logs for evaluation steps.
    """
    def __init__(self, eval_env, save_path, verbose=1):
        super(SaveBestModelOnEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.best_mean_reward = -float("inf")
        self.eval_frequency = 1000  # Define the evaluation frequency internally
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_frequency == 0:
            logger.info(f"Starting evaluation at step {self.n_calls}...")
            episode_rewards = []
            obs = self.eval_env.reset()
            done = False
            step_count = 0

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = self.eval_env.step(action)
                episode_rewards.append(rewards[0])
                step_count += 1
                logger.info(f"Step {step_count}: Reward={rewards[0]}, Done={dones[0]}")
                done = dones[0]

            mean_reward = np.mean(episode_rewards)
            logger.info(f"Evaluation completed: Mean Reward={mean_reward}, Total Steps={step_count}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                save_file = os.path.join(self.save_path, f"best_model_{timestamp}.zip")
                self.model.save(save_file)
                logger.info(f"New best model saved at {save_file} with mean reward: {mean_reward}")
        return True


def hyperparameter_tuning(trial):
    """
    Objective function for hyperparameter tuning with Optuna.
    """
    env = DummyVecEnv([make_env(render_mode="none")])

    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_int("buffer_size", 10000, 30000, step=5000)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    target_update_interval = trial.suggest_int("target_update_interval", 500, 5000, step=500)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=0,
    )

    # Train the model for a fixed number of timesteps
    model.learn(total_timesteps=200000)

    # Evaluate the model
    mean_reward = 0
    obs = env.reset()
    for _ in range(10):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            mean_reward += rewards[0]
            done = dones[0]
    mean_reward /= 10  # Average reward

    env.close()
    return mean_reward


def tune_hyperparameters():
    """
    Run Optuna hyperparameter optimization.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(hyperparameter_tuning, n_trials=OPTUNA_TRIALS)

    print("Best hyperparameters:", study.best_params)
    return study.best_params


def train_agent_with_tuned_params(env, model_path, params, total_timesteps):
    """
    Train the agent using tuned hyperparameters.
    """
    os.makedirs(model_path, exist_ok=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=params["learning_rate"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        train_freq=params["train_freq"],
        target_update_interval=params["target_update_interval"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=params["exploration_final_eps"],
        verbose=1,
        tensorboard_log="./simulated_tensorboard/",
    )

    save_on_step_callback = SaveOnStepCallback(
        save_freq=SAVE_EVERY_STEPS,
        save_path="models/step_checkpoints",
        verbose=1
    )

    save_best_model_callback = SaveBestModelOnEvalCallback(
        eval_env=DummyVecEnv([make_env(render_mode="none")]),
        save_path=os.path.join(model_path, "best_models"),
        verbose=1
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[save_on_step_callback, save_best_model_callback]
    )

    model.save(f"{model_path}/final_model")
    env.close()


def main():
    print("Starting hyperparameter tuning...")
    best_params = tune_hyperparameters()

    print("Training with best parameters...")
    env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])
    train_agent_with_tuned_params(env, MODEL_PATH, best_params, total_timesteps=2000000)


if __name__ == "__main__":
    main()

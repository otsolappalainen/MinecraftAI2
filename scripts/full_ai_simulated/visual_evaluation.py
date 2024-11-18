import os
import logging
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env_simulated import SimulatedEnvGraphics  # Import the environment
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = r"E:\model_spam\best_models"  # Path where models are saved
TRAJECTORY_LOG_PATH = r"E:\model_spam\trajectories"  # Path to save the trajectories

os.makedirs(TRAJECTORY_LOG_PATH, exist_ok=True)  # Ensure the directory exists


def make_env(render_mode="none"):
    """
    Create a simulated environment instance.
    """
    def _init():
        return SimulatedEnvGraphics(render_mode=render_mode)
    return _init


def load_model():
    """
    Ask the user to select a model from the folder and load it.
    """
    models = [f for f in os.listdir(MODEL_PATH) if f.endswith(".zip")]
    if not models:
        logger.error("No models found in the specified folder.")
        exit(1)

    print("Available models:")
    for i, model_name in enumerate(models):
        print(f"{i}: {model_name}")
    model_idx = int(input("Enter the number corresponding to the model you want to load: "))
    model_file = models[model_idx]
    model_path = os.path.join(MODEL_PATH, model_file)
    logger.info(f"Loading model: {model_path}")
    return DQN.load(model_path)


def record_trajectories(model, num_agents, render_mode="none"):
    """
    Run the simulation and record trajectories for multiple agents.
    """
    env = DummyVecEnv([make_env(render_mode=render_mode)])
    all_trajectories = []

    for agent_idx in range(num_agents):
        logger.info(f"Recording trajectory for agent {agent_idx + 1}...")
        obs = env.reset()
        done = [False]
        trajectory = []

        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            if not isinstance(action, (list, np.ndarray)):
                action = np.array([action])
            elif isinstance(action, np.ndarray) and action.ndim == 0:
                action = action.reshape(1)

            obs, rewards, dones, infos = env.step(action)
            done = dones

            # Record trajectory: obs, action, reward
            trajectory.append((obs.copy(), action.copy(), rewards[0]))

        all_trajectories.append(trajectory)

        # Save each trajectory to a file immediately after recording
        trajectory_file = os.path.join(TRAJECTORY_LOG_PATH, f"trajectory_agent_{agent_idx + 1}.pkl")
        with open(trajectory_file, "wb") as file:
            pickle.dump(trajectory, file)

    env.close()
    return all_trajectories


def load_trajectory(file_path):
    """
    Load a trajectory from a file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


def main():
    logger.info("Starting evaluation...")
    model = load_model()

    num_agents = int(input("Enter the number of agents to record: "))
    record_trajectories(model, num_agents, render_mode="none")
    logger.info(f"Trajectories saved in: {TRAJECTORY_LOG_PATH}")


if __name__ == "__main__":
    main()
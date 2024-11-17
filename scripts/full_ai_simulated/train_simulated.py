import os
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_simulated import SimulatedEnvGraphics  # Import the environment

# Hardcoded parameters
TOTAL_TIMESTEPS = 100000  # Total timesteps for training
LEARNING_RATE = 0.001  # Learning rate for training
MODEL_PATH = "./simulated_dqn"  # Path to save the trained model
PARALLEL_ENVS = 1  # Number of parallel environments (set > 1 for parallel mode)
RENDER_MODE = "human"
SAVE_EVERY_EPISODES = 2000  # Save the model every 2000 episodes

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def make_env(render_mode="none"):
    """
    Create a simulated environment instance.
    """
    def _init():
        return SimulatedEnvGraphics(render_mode=render_mode)
    return _init

class SaveOnEpisodeCallback(BaseCallback):
    """
    Custom callback to save the model every N episodes.
    """
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveOnEpisodeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Ensure episode count is available
        if "episodes" in self.locals:
            episodes = self.locals["episodes"]
            if episodes % self.save_freq == 0:
                save_file = os.path.join(self.save_path, f"model_episode_{episodes}.zip")
                self.model.save(save_file)
                if self.verbose > 0:
                    logger.info(f"Model saved at episode {episodes} to {save_file}")
        return True


def train_agent(env, model_path, learning_rate, total_timesteps, parallel_envs):
    """
    Train a DQN agent in the simulated environment.
    """
    # Ensure the model path directory exists
    os.makedirs(model_path, exist_ok=True)

    # Wrap the environment for parallel training if `parallel_envs > 1`
    if parallel_envs > 1:
        env = SubprocVecEnv([make_env(render_mode="none") for _ in range(parallel_envs)])
    else:
        env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])

    logger.info("Initializing the DQN model...")
    # Create the DQN model with a similar structure as the real environment
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
        tensorboard_log="./simulated_tensorboard/",
    )

    # Add the callback for saving the model every N episodes
    save_callback = SaveOnEpisodeCallback(save_freq=SAVE_EVERY_EPISODES, save_path=model_path)

    logger.info("Starting training...")
    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=save_callback)

    # Save the final model for later use
    logger.info(f"Saving trained model to {model_path}/final_model.zip")
    model.save(f"{model_path}/final_model")

    # Close the environment
    env.close()

    return model


def validate_agent(model_path):
    """
    Validate the trained model in graphical mode.
    """
    logger.info("Validating the trained model in graphical mode...")
    env = DummyVecEnv([make_env(render_mode=RENDER_MODE)])

    # Load the trained model
    model = DQN.load(f"{model_path}/final_model")

    # Run a single episode
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    logger.info(f"Validation completed. Total reward: {total_reward}")
    env.close()


def main():
    logger.info("Starting training with the following hardcoded parameters:")
    logger.info(f" - Total timesteps: {TOTAL_TIMESTEPS}")
    logger.info(f" - Learning rate: {LEARNING_RATE}")
    logger.info(f" - Model path: {MODEL_PATH}")
    logger.info(f" - Parallel environments: {PARALLEL_ENVS}")
    logger.info(f" - Render mode (training): {RENDER_MODE}")

    # Train the agent
    model = train_agent(
        env=make_env(render_mode=RENDER_MODE),
        model_path=MODEL_PATH,
        learning_rate=LEARNING_RATE,
        total_timesteps=TOTAL_TIMESTEPS,
        parallel_envs=PARALLEL_ENVS,
    )

    # Validate the agent in graphical mode
    validate_agent(model_path=MODEL_PATH)


if __name__ == "__main__":
    main()

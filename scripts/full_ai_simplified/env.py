import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import csv

# Constants
INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
YAW_RANGE = (-180, 180)
POSITION_RANGE = (-120, 120)
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
YAW_CHANGE = 10  # Degrees to turn left or right
MAX_EPISODE_LENGTH = 500

# Reward Parameters
REWARD_SCALE_POSITIVE = 10
REWARD_SCALE_NEGATIVE = 10
REWARD_PENALTY_STAY_STILL = -5
REWARD_MAX = 10
REWARD_MIN = -10

class SimulatedEnvSimplified(gym.Env):
    """
    Simplified Simulated Environment with limited actions and observation space.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        render_mode="none",
        max_episode_length=MAX_EPISODE_LENGTH,
        log_file="training_data.csv",
        enable_logging=True,
        env_id=0  # Add an environment ID
    ):
        super(SimulatedEnvSimplified, self).__init__()

        # Observation space: x, z, yaw, task_x, task_z
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        # Action space: 0 - Forward, 1 - Backward, 2 - Turn Left, 3 - Turn Right
        self.action_space = spaces.Discrete(24)

        # Simulation parameters
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.hunger = INITIAL_HUNGER
        self.health = INITIAL_HEALTH
        self.alive = INITIAL_ALIVE
        self.cumulative_reward = 0.0
        self.current_task = np.zeros(2, dtype=np.float32)  # Only first two tasks

        self.step_count = 0
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode

        # Logging
        self.enable_logging = enable_logging
        if self.enable_logging:
            # Use a unique log file per environment
            log_file_name = f"training_data_env_{env_id}.csv"
            self.log_file_handle = open(log_file_name, mode="w", newline="")
            self.log_writer = csv.writer(self.log_file_handle)
            self.log_writer.writerow(["episode_id", "step", "x", "z", "yaw", "reward", "task_x", "task_z"])

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _log_step(self, episode_id, step, x, z, yaw, reward, task_x, task_z):
        if self.enable_logging:
            self.log_writer.writerow([episode_id, step, x, z, yaw, reward, task_x, task_z])

    def _get_observation(self):
        return np.array([self.x, self.z, self.yaw, self.current_task[0], self.current_task[1]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print(f"reward: {self.cumulative_reward}")

        self.x = np.random.uniform(POSITION_RANGE[0], POSITION_RANGE[1])
        self.z = np.random.uniform(POSITION_RANGE[0], POSITION_RANGE[1])
        self.yaw = np.random.uniform(YAW_RANGE[0], YAW_RANGE[1])

        self.hunger = INITIAL_HUNGER
        self.health = INITIAL_HEALTH
        self.alive = INITIAL_ALIVE
        self.cumulative_reward = 0.0
        self.step_count = 0

        # Set current task
        possible_tasks = [
            [0, -1],
            [0, 1],
            [1, 0],
            [-1, 0]
        ]
        self.current_task = np.array(random.choice(possible_tasks), dtype=np.float32)

        self._log_step(episode_id=0, step=self.step_count, x=self.x, z=self.z, yaw=self.yaw,
                      reward=0.0, task_x=self.current_task[0], task_z=self.current_task[1])

        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        reward = -0.1  # Step penalty

        prev_x = self.x

        # Update position based on action
        if action == ACTION_MOVE_FORWARD:
            # Move forward based on yaw
            self.x += np.cos(np.radians(self.yaw))  # Use cos for x
            self.z -= np.sin(np.radians(self.yaw))  # Use -sin for z
        elif action == ACTION_MOVE_BACKWARD:
            # Move backward based on yaw
            self.x -= np.cos(np.radians(self.yaw))  # Use cos for x
            self.z += np.sin(np.radians(self.yaw))  # Use -sin for z
        elif action == ACTION_TURN_LEFT:
            # Turn left (counterclockwise)
            self.yaw -= YAW_CHANGE
            # Wrap yaw to [-180, 180)
            self.yaw = (self.yaw + 180) % 360 - 180
        elif action == ACTION_TURN_RIGHT:
            # Turn right (clockwise)
            self.yaw += YAW_CHANGE
            # Wrap yaw to [-180, 180)
            self.yaw = (self.yaw + 180) % 360 - 180
        elif 4 <= action < 24:
            # Placeholder for additional actions
            pass

        # Calculate reward based on progress in the preferred direction (x-axis)
        delta_x = self.x - prev_x
        reward += delta_x  # Positive for moving right, negative for moving left

        # Update cumulative reward
        self.cumulative_reward += reward

        # Check for episode termination
        terminated = self.step_count >= self.max_episode_length
        truncated = False

        # Get observation
        observation = self._get_observation()

        # Log step
        if self.step_count % 5 == 0:
            self._log_step(episode_id=0, step=self.step_count, x=self.x, z=self.z, yaw=self.yaw,
                          reward=reward, task_x=self.current_task[0], task_z=self.current_task[1])

        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass  # No rendering in simplified version

    def close(self):
        if self.enable_logging and self.log_file_handle:
            self.log_file_handle.close()
        super(SimulatedEnvSimplified, self).close()

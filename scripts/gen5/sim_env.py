import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import csv

# Constants
INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
YAW_RANGE = (-180, 180)
POSITION_RANGE = (-120, 120)
ACTION_DURATION = 0.05  # Simulate action duration in seconds
YAW_CHANGE = 10  # Degrees to turn left or right
ACTION_MOVE_SCALE = 2.0  # Scale for movement distance
MAX_EPISODE_LENGTH = 500

# Reward Parameters
REWARD_SCALE_POSITIVE = 10
REWARD_SCALE_NEGATIVE = 10
REWARD_PENALTY_STAY_STILL = -5
REWARD_MAX = 10
REWARD_MIN = -10

class SimulatedEnvSimplified(gym.Env):
    """
    Simplified Simulated Environment with action durations and dynamic updates.
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
        self.current_task = np.zeros(2, dtype=np.float32)  # Task direction

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

    def _simulate_action(self, action_name, duration):
        """Simulate action execution with intermediate state updates."""
        if action_name == "move_forward":
            dx = np.cos(np.radians(self.yaw)) * ACTION_MOVE_SCALE
            dz = -np.sin(np.radians(self.yaw)) * ACTION_MOVE_SCALE
            self.x += dx * duration
            self.z += dz * duration
        elif action_name == "move_backward":
            dx = -np.cos(np.radians(self.yaw)) * ACTION_MOVE_SCALE
            dz = np.sin(np.radians(self.yaw)) * ACTION_MOVE_SCALE
            self.x += dx * duration
            self.z += dz * duration
        elif action_name == "turn_left":
            self.yaw = (self.yaw - YAW_CHANGE * duration + 360) % 360
        elif action_name == "turn_right":
            self.yaw = (self.yaw + YAW_CHANGE * duration + 360) % 360

    def step(self, action):
        self.step_count += 1
        reward = -0.1  # Step penalty

        action_map = {
            0: "move_forward",
            1: "move_backward",
            2: "turn_left",
            3: "turn_right"
        }

        # Simulate action
        if action in action_map:
            start_x, start_z = self.x, self.z
            self._simulate_action(action_map[action], ACTION_DURATION)

            # Calculate distance traveled for reward
            distance = np.sqrt((self.x - start_x) ** 2 + (self.z - start_z) ** 2)
            reward += distance * REWARD_SCALE_POSITIVE

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

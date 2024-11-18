import gymnasium as gym
from gymnasium import spaces
import torch as th
import numpy as np
import random
import pygame
import time
import csv


class SimulatedEnvGraphics(gym.Env):
    """
    Optimized Simulated environment with lightweight graphics for visualization
    and compatibility with the real environment.
    """

    def __init__(
        self,
        render_mode="none",
        grid_size=2000,
        cell_size=50,
        task_size=20,
        max_episode_length=250,
        simulation_speed=5,
        zoom_factor=0.2,
        device="cpu",  # Dynamically set device if not provided
        log_file=r"E:\CNN\training_data.csv",
    ):
        super(SimulatedEnvGraphics, self).__init__()

        

        # Observation space parameters
        self.image_height = 224
        self.image_width = 224
        self.image_channels = 1  # Grayscale image
        self.log_file = log_file

        self.device = "cpu"
        self.agent_id="agent_1",
        self.position_size = 5  # x, y, z, yaw, pitch
        self.scalar_size = 3  # health, hunger, alive
        self.task_size = task_size
        self.other_size = self.position_size + self.scalar_size + self.task_size

        # Define observation space using Dict for image and other data
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(self.image_channels, self.image_height, self.image_width), dtype=np.float32
                ),
                "other": spaces.Box(low=-np.inf, high=np.inf, shape=(self.other_size,), dtype=np.float32),
            }
        )

        # Action space
        self.action_space = spaces.Discrete(25)  # Placeholder for future use

        # Simulation parameters
        self.x = 0
        self.z = 0
        self.yaw = 0
        self.pitch = 0
        self.hunger = 100
        self.health = 100
        self.alive = 1
        self.cumulative_reward = 0
        self.current_task = np.zeros(task_size, dtype=np.float32)
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.zoom_factor = zoom_factor

        self.episode_id = 0
        self.step_count = 0
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode
        self.simulation_speed = simulation_speed  # Updates per second
        self.time_step = 1.0 / self.simulation_speed  # Time interval per update
        self.window_size = min(grid_size, 1000)  # Limit rendering window size to 1000x1000
        self.window = None
        self.clock = None

        # Initialize the log file
        with open(self.log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "episode_id", "step", "x", "z", "yaw", "reward", "task_x", "task_z"])


    def _log_step(self, x, z, yaw, reward, task_x, task_z):
        """
        Log data for the current step.
        """
        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.agent_id, self.episode_id, self.step_count, x, z, yaw, reward, task_x, task_z])


    def _get_observation(self):
        """
        Create an observation with image and other data.
        Returns PyTorch tensors for Stable-Baselines3 compatibility.
        """
        # Generate a blank grayscale image (224x224)
        image = np.full(
            (self.image_channels, self.image_height, self.image_width),
            128 / 255.0,
            dtype=np.float32,
        )
        # Add noise to the image
        image += np.random.normal(0, 0.01, size=image.shape)
        image = np.clip(image, 0, 1)

        # Positional and scalar values
        positional_values = np.array(
            [self.x, 0, self.z, self.yaw, self.pitch], dtype=np.float32
        )
        scalar_values = np.array(
            [self.hunger, self.health, self.alive], dtype=np.float32
        )
        task_values = np.array(self.current_task, dtype=np.float32)

        # Combine into 'other' observation
        other_observation = np.concatenate([positional_values, scalar_values, task_values])

        # Convert to tensors on the appropriate device
        return {
            "image": th.tensor(image, dtype=th.float32, device=self.device),
            "other": th.tensor(other_observation, dtype=th.float32, device=self.device),
    }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        if self.cumulative_reward != 0:
            print(f"Episode reward: {self.cumulative_reward}")

        # Reset simulation parameters
        self.x = np.random.uniform(-120, 120)
        self.z = np.random.uniform(-120, 120)

        self.yaw = np.random.uniform(-180, 180)  # Random yaw (-180 to 180 degrees)
        self.pitch = np.random.uniform(-90, 90)  # Random pitch (-90 to 90 degrees)

        self.hunger = 100
        self.health = 100
        self.alive = 1
        self.step_count = 0
        self.prev_x = self.x  # Initialize prev_x to the starting x position
        self.prev_z = self.z  # Initialize prev_z to the starting z position
        self.cumulative_reward = 0

        # Set current task with at least one of the first two positions being 1
        self.current_task = np.zeros(self.task_size, dtype=np.float32)
        possible_tasks = [
            [-1, 0],
            #[1, 0],
        ]


        self.current_task[:2] = np.array(random.choice(possible_tasks), dtype=np.float32)

        #print(f"Current task: {self.current_task[:2]}")
        self._log_step(self.x, self.z, self.yaw, 0, self.current_task[0], self.current_task[1])


        observation = self._get_observation()
        return observation, {}


    def step(self, action):
        """
        Perform a step in the environment.
        """
        if not hasattr(self, "no_progress_count"):
            self.no_progress_count = 0

        self.step_count += 1

        # Actions (only a subset affects the simulation)
        if action == 0:  # Move forward
            self.x += np.sin(np.radians(self.yaw))
            self.z += np.cos(np.radians(self.yaw))
        elif action == 1:  # Move backward
            self.x -= np.sin(np.radians(self.yaw))
            self.z -= np.cos(np.radians(self.yaw))
        elif action == 2:  # Turn left
            self.yaw = (self.yaw - 10) % 360 - 180
        elif action == 3:  # Turn right
            self.yaw = (self.yaw + 10) % 360 - 180

        # Calculate delta_x and delta_z
        delta_x = self.x - self.prev_x
        delta_z = self.z - self.prev_z

        # Update previous positions
        self.prev_x = self.x
        self.prev_z = self.z

        # Calculate reward based on the current task
        current_task_vector = th.tensor(self.current_task[:2], dtype=th.float32, device=self.device)
        movement_vector = th.tensor([delta_x, delta_z], dtype=th.float32, device=self.device)
        dot_product = th.dot(movement_vector, current_task_vector).item()

        if dot_product > 0:
            # Reward for moving in the correct direction
            reward = dot_product * 10 + 5  # Bonus reward
            self.no_progress_count = 0  # Reset the no progress counter
        elif dot_product < 0:
            # Penalty for moving in the wrong direction
            reward = dot_product * 10 - 5
            self.no_progress_count += 1
        else:
            # Strong penalty for staying still
            self.no_progress_count += 1
            reward = -5

        # Add escalating penalty for staying still or oscillating
        reward -= self.no_progress_count ** 2 * 0.1

        # Clamp reward to prevent runaway values (optional)
        reward = max(reward, -50)

        # Update cumulative reward
        self.cumulative_reward += reward

        # Determine if the episode is done
        terminated = self.step_count >= self.max_episode_length
        truncated = False

        # Get observation
        observation = self._get_observation()

        

        if self.step_count % 5 == 0:
            self._log_step(self.x, self.z, self.yaw, reward, self.current_task[0], self.current_task[1])
            #print(f"x: {self.x} z: {self.z} yaw: {self.yaw} CR: {self.cumulative_reward}")


        return observation, reward, terminated, truncated, {}

    # Remaining methods (_initialize_graphics, _render, close) are unchanged

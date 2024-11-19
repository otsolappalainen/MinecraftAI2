import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import csv

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 1  # Grayscale image
TASK_SIZE = 20
ACTION_SPACE_SIZE = 25
MAX_EPISODE_LENGTH = 500
DEVICE = "cpu"

INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
YAW_RANGE = (-180, 180)
PITCH_RANGE = (-90, 90)
POSITION_RANGE = (-120, 120)

ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
YAW_CHANGE = 10  # Degrees to turn left or right

REWARD_SCALE_POSITIVE = 10
REWARD_SCALE_NEGATIVE = 5
REWARD_PENALTY_STAY_STILL = -5
REWARD_MAX = 10
REWARD_MIN = -10

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
        task_size=TASK_SIZE,
        max_episode_length=MAX_EPISODE_LENGTH,
        simulation_speed=5,
        zoom_factor=0.2,
        device=DEVICE,
        log_file=r"E:\CNN\training_data.csv",
        enable_logging=True,
    ):
        super(SimulatedEnvGraphics, self).__init__()

        # Observation space parameters
        self.image_height = IMAGE_HEIGHT
        self.image_width = IMAGE_WIDTH
        self.image_channels = IMAGE_CHANNELS
        self.log_file = log_file
        self.enable_logging = enable_logging

        self.device = device
        self.agent_id = "agent_1"
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
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)  # Placeholder for future use

        # Simulation parameters
        self.x = 0
        self.z = 0
        self.yaw = 0
        self.pitch = 0
        self.hunger = INITIAL_HUNGER
        self.health = INITIAL_HEALTH
        self.alive = INITIAL_ALIVE
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


        # Open the log file once during initialization
        self.log_file_handle = open(self.log_file, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file_handle)
        # Write the header row
        self.log_writer.writerow(["agent_id", "episode_id", "step", "x", "z", "yaw", "reward", "task_x", "task_z"])
        

        # Cache the constant image
        self.constant_image = np.full((self.image_channels, self.image_height, self.image_width), 128 / 255.0, dtype=np.float32)

    def _log_step(self, x, z, yaw, reward, task_x, task_z):
        """
        Log data for the current step.
        """
        self.log_writer.writerow([self.agent_id, self.episode_id, self.step_count, x, z, yaw, reward, task_x, task_z])


    def seed(self, seed=None):
        # Set the seed for this env's random number generator(s).
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    


    def _get_observation(self):
        """
        Create an observation with image and other data.
        """
        image = self.constant_image  # Use cached image
        positional_values = np.array([self.x, 0, self.z, self.yaw, self.pitch], dtype=np.float32)
        scalar_values = np.array([self.hunger, self.health, self.alive], dtype=np.float32)
        task_values = np.array(self.current_task, dtype=np.float32)

        other_observation = np.concatenate([positional_values, scalar_values, task_values])

        return {"image": image, "other": other_observation}

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)
        if self.cumulative_reward != 0:
            print(f"Episode reward: {self.cumulative_reward}")

        # Reset simulation parameters
        self.x = np.random.uniform(POSITION_RANGE[0], POSITION_RANGE[1])
        self.z = np.random.uniform(POSITION_RANGE[0], POSITION_RANGE[1])

        self.yaw = np.random.uniform(YAW_RANGE[0], YAW_RANGE[1])  # Random yaw (-180 to 180 degrees)
        self.pitch = np.random.uniform(PITCH_RANGE[0], PITCH_RANGE[1])  # Random pitch (-90 to 90 degrees)

        self.hunger = INITIAL_HUNGER
        self.health = INITIAL_HEALTH
        self.alive = INITIAL_ALIVE
        self.step_count = 0
        self.prev_x = self.x  # Initialize prev_x to the starting x position
        self.prev_z = self.z  # Initialize prev_z to the starting z position
        self.cumulative_reward = 0

        # Set current task
        self.current_task = np.zeros(self.task_size, dtype=np.float32)
        possible_tasks = [
            [0, -1],
            [0, 1],
        ]
        self.current_task[:2] = np.array(random.choice(possible_tasks), dtype=np.float32)

        self._log_step(self.x, self.z, self.yaw, 0, self.current_task[0], self.current_task[1])

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """
        Perform a step in the environment.
        """
        self.step_count += 1

        # Actions
        if action == ACTION_MOVE_FORWARD:  # Move forward
            self.x += np.sin(np.radians(self.yaw))
            self.z += np.cos(np.radians(self.yaw))
        elif action == ACTION_MOVE_BACKWARD:  # Move backward
            self.x -= np.sin(np.radians(self.yaw))
            self.z -= np.cos(np.radians(self.yaw))
        elif action == ACTION_TURN_LEFT:  # Turn left
            self.yaw = (self.yaw - YAW_CHANGE) % 360 - 180
        elif action == ACTION_TURN_RIGHT:  # Turn right
            self.yaw = (self.yaw + YAW_CHANGE) % 360 - 180

        # Calculate delta_x and delta_z
        delta_x = self.x - self.prev_x
        delta_z = self.z - self.prev_z

        # Update previous positions
        self.prev_x = self.x
        self.prev_z = self.z

        # Calculate reward
        current_task_vector = self.current_task[:2]
        movement_vector = np.array([delta_x, delta_z], dtype=np.float32)
        dot_product = np.dot(movement_vector, current_task_vector)

        if dot_product > 0:
            reward = min(dot_product * REWARD_SCALE_POSITIVE, REWARD_MAX)
        elif dot_product < 0:
            reward = max(dot_product * REWARD_SCALE_NEGATIVE, REWARD_MIN)
        else:
            reward = REWARD_PENALTY_STAY_STILL

        # Update cumulative reward
        self.cumulative_reward += reward

        # Determine if the episode is done
        terminated = self.step_count >= self.max_episode_length
        truncated = False

        # Get observation
        observation = self._get_observation()

        # Log step if needed
        if self.step_count % 25 == 0:
            self._log_step(self.x, self.z, self.yaw, reward, self.current_task[0], self.current_task[1])

        return observation, reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass  # Implement rendering if needed

    def close(self):
        """
        Close the environment and any open resources.
        """
        if self.log_file_handle:
            self.log_file_handle.close()  # Ensure the file handle is properly closed
        super(SimulatedEnvGraphics, self).close()
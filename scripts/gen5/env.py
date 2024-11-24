import websocket
import json
import numpy as np
import random
import csv
from gymnasium import Env, spaces

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 1
TASK_SIZE = 20
ACTION_SPACE_SIZE = 25
MAX_EPISODE_LENGTH = 500
DEVICE = "cpu"
WEBSOCKET_URL = "ws://localhost:8080"  # URL for WebSocket server

INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
POSITION_RANGE = (-120, 120)

class SimulatedEnvGraphics(Env):
    def __init__(
        self,
        render_mode="none",
        task_size=TASK_SIZE,
        max_episode_length=MAX_EPISODE_LENGTH,
        log_file="training_data.csv",
        enable_logging=True,
    ):
        super(SimulatedEnvGraphics, self).__init__()

        self.render_mode = render_mode
        self.task_size = task_size
        self.max_episode_length = max_episode_length
        self.log_file = log_file
        self.enable_logging = enable_logging

        # Observation and action spaces
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=1, shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32
            ),
            "other": spaces.Box(low=-np.inf, high=np.inf, shape=(5 + 3 + task_size,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Environment variables
        self.current_task = np.zeros(task_size, dtype=np.float32)
        self.episode_id = 0
        self.step_count = 0
        self.cumulative_reward = 0
        self.ws = None  # WebSocket connection

        # Logging setup
        if self.enable_logging:
            self.log_file_handle = open(self.log_file, mode="w", newline="")
            self.log_writer = csv.writer(self.log_file_handle)
            self.log_writer.writerow(["episode_id", "step", "x", "z", "yaw", "reward", "task_x", "task_z"])

    def connect_websocket(self):
        """Establish WebSocket connection."""
        self.ws = websocket.WebSocket()
        self.ws.connect(WEBSOCKET_URL)

    def send_action(self, action_name):
        """Send an action to the mod and return the response."""
        action_packet = {"action": action_name}
        self.ws.send(json.dumps(action_packet))
        response = self.ws.recv()
        return json.loads(response)

    def _log_step(self, x, z, yaw, reward, task_x, task_z):
        """Log data for the current step."""
        if self.enable_logging:
            self.log_writer.writerow([self.episode_id, self.step_count, x, z, yaw, reward, task_x, task_z])

    def _get_observation(self, state):
        """Generate an observation based on the state returned by the mod."""
        # Simulate a static image (replace with real image capture if available)
        image = np.full((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), 128 / 255.0, dtype=np.float32)

        # Parse state data
        positional_values = np.array([state["x"], 0, state["z"], state["yaw"], state["pitch"]], dtype=np.float32)
        scalar_values = np.array([state["hunger"], state["health"], state["alive"]], dtype=np.float32)
        task_values = np.array(self.current_task, dtype=np.float32)

        other_observation = np.concatenate([positional_values, scalar_values, task_values])

        return {"image": image, "other": other_observation}

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        if self.cumulative_reward != 0:
            print(f"Episode reward: {self.cumulative_reward}")

        # Reset environment variables
        self.episode_id += 1
        self.step_count = 0
        self.cumulative_reward = 0
        self.current_task = np.zeros(self.task_size, dtype=np.float32)
        possible_tasks = [[0, -1], [0, 1]]
        self.current_task[:2] = np.array(random.choice(possible_tasks), dtype=np.float32)

        # Establish WebSocket connection
        if not self.ws or not self.ws.connected:
            self.connect_websocket()

        # Send reset command to the mod
        state = self.send_action("reset")
        observation = self._get_observation(state)

        # Log the reset state
        self._log_step(state["x"], state["z"], state["yaw"], 0, self.current_task[0], self.current_task[1])
        return observation, {}

    def step(self, action):
        """Perform a step in the environment."""
        self.step_count += 1

        # Map action index to action names
        actions = ["move_forward", "move_backward", "turn_left", "turn_right"]
        action_name = actions[action]

        # Send action and get the new state
        state = self.send_action(action_name)

        # Compute reward
        delta_x = state["x"] - self.current_task[0]
        delta_z = state["z"] - self.current_task[1]
        reward = -np.sqrt(delta_x**2 + delta_z**2)  # Example reward function

        self.cumulative_reward += reward

        # Check termination conditions
        terminated = self.step_count >= self.max_episode_length
        truncated = False

        # Generate observation
        observation = self._get_observation(state)

        # Log the step
        self._log_step(state["x"], state["z"], state["yaw"], reward, self.current_task[0], self.current_task[1])

        return observation, reward, terminated, truncated, {}

    def close(self):
        """Close the WebSocket connection and log file."""
        if self.ws:
            self.ws.close()
        if self.enable_logging and hasattr(self, 'log_file_handle') and self.log_file_handle:
            self.log_file_handle.close()
        super(SimulatedEnvGraphics, self).close()

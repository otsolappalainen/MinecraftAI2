import websocket
import json
import numpy as np
import random
import csv
import threading
import time
import cv2
import pygetwindow as gw
from mss import mss
from gymnasium import Env, spaces

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 1
TASK_SIZE = 20
ACTION_SPACE_SIZE = 17  # Updated to 17
MAX_EPISODE_LENGTH = 500
DEVICE = "cpu"
WEBSOCKET_URL = "ws://localhost:8080"  # URL for WebSocket server

INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
YAW_RANGE = (-180, 180)
PITCH_RANGE = (-90, 90)
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

        self.render_mode = "none"
        self.task_size = task_size
        self.max_episode_length = max_episode_length
        self.log_file = log_file
        self.enable_logging = enable_logging

        # Observation and action spaces
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=1, shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32
            ),
            "other": spaces.Box(low=-np.inf, high=np.inf, shape=(9 + task_size,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Environment variables
        self.current_task = np.zeros(task_size, dtype=np.float32)
        self.episode_id = 0
        self.step_count = 0
        self.cumulative_reward = 0
        self.ws = None  # WebSocket connection
        self.image = None  # Captured image

        # Initial state variables
        self.initial_state = None

        # Logging setup
        if self.enable_logging:
            self.log_file_handle = open(self.log_file, mode="w", newline="")
            self.log_writer = csv.writer(self.log_file_handle)
            self.log_writer.writerow(["episode_id", "step", "x", "y", "z", "yaw", "pitch", "reward", "task_x", "task_z"])

        # Cached constant image
        self.constant_image = np.full((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), 128 / 255.0, dtype=np.float32)
        # Initialize RNG
        self.np_random, self.seed_val = None, None

    def connect_websocket(self):
        """Establish WebSocket connection."""
        self.ws = websocket.WebSocket()
        self.ws.connect(WEBSOCKET_URL)

    def send_action(self, action_name, reset_type=None, reset_coordinates=None):
        """Send an action to the mod and return the response."""
        action_packet = {"action": action_name}
        if reset_type is not None:
            action_packet["reset_type"] = reset_type
            if reset_coordinates:
                action_packet["coordinates"] = reset_coordinates
        
        self.ws.send(json.dumps(action_packet))

        # Start capturing the screen in a separate thread while waiting for response
        capture_thread = threading.Thread(target=self.capture_image)
        capture_thread.start()

        response = self.ws.recv()
        capture_thread.join()
        return json.loads(response)

    def capture_image(self):
        """Capture the Minecraft window using MSS and process it."""
        try:
            # Get Minecraft window by title
            window = gw.getWindowsWithTitle('Minecraft')[0]  # Assuming Minecraft is in the title

            # Get the window's bounding box
            left, top, width, height = window.left, window.top, window.width, window.height
            capture_region = {"top": top, "left": left, "width": width, "height": height}

            # Capture the specific window
            with mss() as sct:
                screenshot = sct.grab(capture_region)
                image = np.array(screenshot)
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
                image = image / 255.0  # Normalize to [0, 1]
                self.image = image.reshape(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
        except IndexError:
            print("Minecraft window not found.")
            self.image = self.constant_image

    def _log_step(self, x, y, z, yaw, pitch, reward, task_x, task_z):
        """Log data for the current step."""
        if self.enable_logging:
            self.log_writer.writerow([self.episode_id, self.step_count, x, y, z, yaw, pitch, reward, task_x, task_z])

    def _get_observation(self, state):
        """Generate an observation based on the state returned by the mod."""
        # Use captured image if available, otherwise use cached constant image
        image = self.image if self.image is not None else self.constant_image

        # Parse state data
        positional_values = np.array([state["x"], state["y"], state["z"], state["yaw"], state["pitch"]], dtype=np.float32)
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
        possible_tasks = [[0, -1], [0, 1], [1, 0], [-1, 0]]
        self.current_task[:2] = np.array(random.choice(possible_tasks), dtype=np.float32)

        # Establish WebSocket connection
        if not self.ws or not self.ws.connected:
            self.connect_websocket()

        # Determine reset type
        reset_type = options.get("reset_type", 0) if options else 0
        reset_coordinates = None
        if reset_type in [1, 2] and self.initial_state is None:
            state = self.send_action("get_state")
            self.initial_state = {
                "x": state["x"],
                "y": state["y"],
                "z": state["z"],
                "yaw": state["yaw"],
                "pitch": state["pitch"]
            }

        if reset_type in [1, 2]:
            reset_coordinates = self.initial_state
            if reset_type == 2:
                reset_coordinates["yaw"] = random.uniform(*YAW_RANGE)
                reset_coordinates["pitch"] = random.uniform(*PITCH_RANGE)

        # Send reset command to the mod
        state = self.send_action("reset", reset_type=reset_type, reset_coordinates=reset_coordinates)
        observation = self._get_observation(state)

        # Log the reset state
        self._log_step(state["x"], state["y"], state["z"], state["yaw"], state["pitch"], 0, self.current_task[0], self.current_task[1])
        return observation, {}

    def step(self, action):
        """Perform a step in the environment."""
        self.step_count += 1

        # Map action index to action names
        actions = [
            "move_forward", "move_backward", "turn_left", "turn_right",
            "move_left", "move_right", "sneak_toggle", "no_op"
        ]
        action_name = actions[action] if action < len(actions) else "no_op"

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
        self._log_step(state["x"], state["y"], state["z"], state["yaw"], state["pitch"], reward, self.current_task[0], self.current_task[1])

        return observation, reward, terminated, truncated, {}

    def close(self):
        """Close the WebSocket connection and log file."""
        if self.ws:
            self.ws.close()
        if self.enable_logging and hasattr(self, 'log_file_handle') and self.log_file_handle:
            self.log_file_handle.close()
        super(SimulatedEnvGraphics, self).close()

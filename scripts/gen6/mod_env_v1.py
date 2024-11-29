# File: mod_env_v1.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import websockets
import json
import time
import cv2
from mss import mss
from PIL import Image
import pygetwindow as gw
import logging
from datetime import datetime
import math

# Constants
TASK_SIZE = 20
ACTION_SPACE_SIZE = 18  # Total number of actions (0-17)
MAX_EPISODE_LENGTH = 500

# Yaw Range in degrees
YAW_RANGE = (-180, 180)

# Image Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3  # RGB

# Define the global blank image as a constant
# Using a neutral gray color (128/255) for each pixel in RGB
BLANK_IMAGE = np.full(
    (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
    128 / 255.0,
    dtype=np.float32
)

# Action Definitions
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_MOVE_LEFT = 4
ACTION_MOVE_RIGHT = 5
# Actions 6-17 are unused

# Reward Definitions
REWARD_SCALE_POSITIVE = 2.0  # Reward multiplier for positive movement
REWARD_PENALTY_STAY_STILL = -0.1  # Penalty for non-movement

# Movement Parameters
MOVE_DISTANCE_PER_STEP = 0.5  # Desired position change per timestep
TURN_ANGLE_PER_STEP = 15  # Degrees to turn per step

class MinecraftEnv(gym.Env):
    """
    Gymnasium environment for interacting with Minecraft mod via WebSockets.
    Sends actions to the mod, captures screenshots, processes images,
    and returns observations.
    """

    metadata = {"render_modes": ["human"]}

    # Hardcoded debug flag
    DEBUG = True

    def __init__(self, ws_uri="ws://localhost:8080"):
        super(MinecraftEnv, self).__init__()

        # Initialize logging
        self._init_logging()

        # WebSocket URI
        self.ws_uri = ws_uri

        # Initialize WebSocket connection
        self.ws = None
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self._connect())

        # Define action and observation spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=BLANK_IMAGE.shape,  # (3, 224, 224)
                dtype=np.float32
            ),
            "other": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(8 + TASK_SIZE,),  # 8 state variables + 20 task variables
                dtype=np.float32,
            )
        })

        # Define the region of the Minecraft window for screenshot
        self.region = self._get_minecraft_window_region()

    def _init_logging(self):
        """Initialize logging based on the DEBUG flag."""
        self.logger = logging.getLogger("MinecraftEnv")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.DEBUG:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    async def _connect(self):
        """Establish WebSocket connection."""
        try:
            self.ws = await websockets.connect(self.ws_uri)
            self.logger.debug(f"Connected to WebSocket server at {self.ws_uri}")
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket server: {e}")
            raise e

    def _get_minecraft_window_region(self):
        """Retrieve the position and size of the Minecraft window."""
        try:
            windows = gw.getWindowsWithTitle('Minecraft')
            if not windows:
                self.logger.error("Minecraft window not found. Ensure Minecraft is running.")
                raise Exception("Minecraft window not found.")
            minecraft_window = windows[0]
            if not minecraft_window.visible:
                self.logger.warning("Minecraft window is not visible. It might be minimized.")
            region = {
                "top": minecraft_window.top,
                "left": minecraft_window.left,
                "width": minecraft_window.width,
                "height": minecraft_window.height
            }
            self.logger.debug(f"Minecraft window region: {region}")
            return region
        except Exception as e:
            self.logger.error(f"Error retrieving Minecraft window region: {e}")
            raise e

    def _capture_screenshot(self):
        """
        Capture the screenshot of the Minecraft window.
        Ensures that the screenshot is taken even if the window is in the background.
        """
        try:
            with mss() as sct:
                screenshot = sct.grab(self.region)
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                img = np.array(img)
                self.logger.debug("Screenshot captured.")
                return img
        except Exception as e:
            self.logger.error(f"Error capturing screenshot: {e}")
            return np.zeros(self.observation_space["image"].shape, dtype=np.float32)

    def _process_image(self, image):
        """
        Process the captured image:
        - Resize to the desired dimensions.
        - Normalize pixel values to [0, 1].
        - Transpose to channels-first format (3, 224, 224).
        """
        try:
            resized_image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # (224, 224, 3)
            normalized_image = resized_image.astype(np.float32) / 255.0  # (224, 224, 3)
            # Transpose to (3, 224, 224)
            normalized_image = np.transpose(normalized_image, (2, 0, 1))
            self.logger.debug("Image processed (resized, normalized, and transposed).")
            return normalized_image
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return np.zeros(self.observation_space["image"].shape, dtype=np.float32)

    def _get_action_mapping(self):
        """Map action indices to action names as per the updated action list."""
        return {
            0: "move_forward",
            1: "move_backward",
            2: "move_left",
            3: "move_right",
            4: "jump_walk_forward",
            5: "jump",
            6: "sneak_toggle",
            7: "look_left",
            8: "look_right",
            9: "look_up",
            10: "look_down",
            11: "big_turn_left",
            12: "big_turn_right",
            13: "mouse_left_click",
            14: "mouse_right_click",
            15: "next_item",
            16: "previous_item",
            17: "no_op"  # Define action 17 as 'no_op' to handle unused actions
        }

    def _build_observation(self, image, state):
        """Construct the observation dictionary with flattened keys."""
        processed_image = self._process_image(image)

        # Normalize x, z, and y coordinates
        normalized_x = (2 * (state.get("x", 0.0) + 20000) / 40000) - 1
        normalized_z = (2 * (state.get("z", 0.0) + 20000) / 40000) - 1
        normalized_y = (2 * (state.get("y", 0.0) + 256) / 512) - 1

        # Encode yaw using sine and cosine
        yaw_rad = math.radians(state.get("yaw", 0.0))
        sin_yaw = math.sin(yaw_rad)
        cos_yaw = math.cos(yaw_rad)

        # Normalize health and hunger (range [0, 20])
        normalized_health = state.get("health", 20.0) / 20.0
        normalized_hunger = state.get("hunger", 20.0) / 20.0

        # Alive status
        alive = float(state.get("alive", True))  # Convert to float for consistency

        # Task vector (assuming it's part of the state; adjust if necessary)
        # For simplicity, we'll initialize it as zeros here
        # Replace this with actual task data if available
        task_vector = np.zeros(TASK_SIZE, dtype=np.float32)
        # If task data is part of the state, extract it here
        # Example:
        # task_vector = np.array(state.get("task", [0.0]*TASK_SIZE), dtype=np.float32)

        # Combine all scalar values into 'other'
        other = np.array(
            [
                normalized_x,
                normalized_z,
                normalized_y,
                sin_yaw,
                cos_yaw,
                normalized_health,
                normalized_hunger,
                alive
            ] + task_vector.tolist(),
            dtype=np.float32
        )

        observation = {
            "image": processed_image,  # Shape: (3, 224, 224)
            "other": other  # Shape: (28,)
        }

        # Log observation shapes
        self.logger.debug(f"Observation shapes: image={observation['image'].shape}, other={observation['other'].shape}")

        return observation

    def _send_action(self, action):
        """Send action to the Minecraft mod via WebSocket."""
        try:
            action_mapping = self._get_action_mapping()
            action_name = action_mapping.get(action, "no_op")
            action_message = json.dumps({"action": action_name})
            self.loop.run_until_complete(self.ws.send(action_message))
            self.logger.debug(f"Sent action: {action_name}")
        except Exception as e:
            self.logger.error(f"Error sending action via WebSocket: {e}")

    def _receive_state(self):
        """Receive state from the Minecraft mod via WebSocket."""
        try:
            state_message = self.loop.run_until_complete(self.ws.recv())
            state = json.loads(state_message)
            self.logger.debug(f"Received state: {state}")
            return state
        except Exception as e:
            self.logger.error(f"Error receiving state via WebSocket: {e}")
            # Return a default state if receiving fails
            return {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "health": 20.0,
                "hunger": 20.0,
                "alive": True,
                "inventory": {str(i): 0 for i in range(9)}
            }

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (int): An action provided by the agent.

        Returns:
            observation (dict): Agent's observation of the current environment.
            reward (float): Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Diagnostic information useful for debugging.
        """
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        start_time = time.time()
        if self.DEBUG:
            self.logger.debug(f"Action {action} started at {datetime.now()}")

        # Send action to the mod
        self._send_action(action)

        # Receive state from the mod
        state = self._receive_state()

        end_time = time.time()
        if self.DEBUG:
            self.logger.debug(f"Action {action} ended at {datetime.now()} (Duration: {end_time - start_time:.4f}s)")

        # Capture screenshot
        image = self._capture_screenshot()

        # Build observation
        observation = self._build_observation(image, state)

        # Define reward (Example: based on health and hunger)
        # You can customize the reward function based on your specific needs
        reward = (state.get("health", 20.0) + state.get("hunger", 20.0)) - 40.0  # Simple example

        # Define termination condition
        terminated = not state.get("alive", True)
        truncated = False  # You can set conditions for truncation if needed

        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation and info.

        Args:
            seed (int, optional): The seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting.

        Returns:
            observation (dict): The initial observation.
            info (dict): Additional information.
        """
        super().reset(seed=seed)

        if self.DEBUG:
            self.logger.debug(f"Resetting environment at {datetime.now()}")

        # Optionally send a reset action to the mod
        self._send_action(17)  # "no_op"

        # Receive the initial state
        state = self._receive_state()

        # Capture initial screenshot
        image = self._capture_screenshot()

        # Build initial observation
        observation = self._build_observation(image, state)

        info = {}

        return observation, info

    def render(self):
        """Render the environment. Not implemented."""
        pass

    def close(self):
        """Close the WebSocket connection and perform cleanup."""
        try:
            if self.ws:
                self.loop.run_until_complete(self.ws.close())
                self.logger.debug("WebSocket connection closed.")
        except Exception as e:
            self.logger.error(f"Error closing WebSocket connection: {e}")
        super(MinecraftEnv, self).close()

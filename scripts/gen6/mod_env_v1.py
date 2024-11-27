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

class MinecraftEnv(gym.Env):
    """
    Gym environment for interacting with Minecraft mod via WebSockets.
    Sends actions to the mod, captures screenshots, processes images,
    and returns observations.
    """

    metadata = {"render.modes": ["human"]}

    # Hardcoded debug flag
    DEBUG = True

    # Constants
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNELS = 3  # RGB

    ACTION_SPACE_SIZE = 20  # Total number of actions (0-19)
    OBSERVATION_IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

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
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0,
                high=1.0,
                shape=self.OBSERVATION_IMAGE_SHAPE,
                dtype=np.float32
            ),
            "state": spaces.Dict({
                "x": spaces.Box(low=-20000.0, high=20000.0, shape=(1,), dtype=np.float32),
                "y": spaces.Box(low=-256.0, high=256.0, shape=(1,), dtype=np.float32),
                "z": spaces.Box(low=-20000.0, high=20000.0, shape=(1,), dtype=np.float32),
                "yaw": spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                "pitch": spaces.Box(low=-90.0, high=90.0, shape=(1,), dtype=np.float32),
                "health": spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32),
                "hunger": spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32),
                "alive": spaces.Discrete(2),  # 1 for alive, 0 for dead
                "inventory": spaces.Dict({
                    str(i): spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32)
                    for i in range(9)  # Hotbar slots 0-8
                })
            })
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
            if not minecraft_window.isVisible:
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
            return np.zeros(self.OBSERVATION_IMAGE_SHAPE, dtype=np.float32)

    def _process_image(self, image):
        """
        Process the captured image:
        - Resize to the desired dimensions.
        - Normalize pixel values to [0, 1].
        """
        try:
            resized_image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
            normalized_image = resized_image.astype(np.float32) / 255.0
            self.logger.debug("Image processed (resized and normalized).")
            return normalized_image
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return np.zeros(self.OBSERVATION_IMAGE_SHAPE, dtype=np.float32)

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
            17: "reset 0",
            18: "reset 1",
            19: "reset 2",
            # Add more mappings if necessary
        }

    def _build_observation(self, image, state):
        """Construct the observation dictionary."""
        processed_image = self._process_image(image)
        # Normalize and format state
        obs_state = {
            "x": np.array([state.get("x", 0.0)], dtype=np.float32),
            "y": np.array([state.get("y", 0.0)], dtype=np.float32),
            "z": np.array([state.get("z", 0.0)], dtype=np.float32),
            "yaw": np.array([state.get("yaw", 0.0)], dtype=np.float32),
            "pitch": np.array([state.get("pitch", 0.0)], dtype=np.float32),
            "health": np.array([state.get("health", 20.0)], dtype=np.float32),
            "hunger": np.array([state.get("hunger", 20.0)], dtype=np.float32),
            "alive": int(state.get("alive", True)),
            "inventory": {
                str(i): np.array([state.get("inventory", {}).get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
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
                "hunger": 20,
                "alive": True,
                "inventory": {str(i): 0 for i in range(9)}
            }

    def _build_observation_with_image(self, image, state):
        """Build the observation dictionary including the image."""
        processed_image = self._process_image(image)
        obs_state = {
            "x": np.array([state.get("x", 0.0)], dtype=np.float32),
            "y": np.array([state.get("y", 0.0)], dtype=np.float32),
            "z": np.array([state.get("z", 0.0)], dtype=np.float32),
            "yaw": np.array([state.get("yaw", 0.0)], dtype=np.float32),
            "pitch": np.array([state.get("pitch", 0.0)], dtype=np.float32),
            "health": np.array([state.get("health", 20.0)], dtype=np.float32),
            "hunger": np.array([state.get("hunger", 20.0)], dtype=np.float32),
            "alive": int(state.get("alive", True)),
            "inventory": {
                str(i): np.array([state.get("inventory", {}).get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
        return observation

    def _build_observation_from_state(self, image, state):
        """Alternative method to build observation."""
        processed_image = self._process_image(image)
        obs_state = {
            "x": np.array([state.get("x", 0.0)], dtype=np.float32),
            "y": np.array([state.get("y", 0.0)], dtype=np.float32),
            "z": np.array([state.get("z", 0.0)], dtype=np.float32),
            "yaw": np.array([state.get("yaw", 0.0)], dtype=np.float32),
            "pitch": np.array([state.get("pitch", 0.0)], dtype=np.float32),
            "health": np.array([state.get("health", 20.0)], dtype=np.float32),
            "hunger": np.array([state.get("hunger", 20.0)], dtype=np.float32),
            "alive": int(state.get("alive", True)),
            "inventory": {
                str(i): np.array([state.get("inventory", {}).get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
        return observation

    def _build_observation_v2(self, image, state):
        """Another variant to build observation."""
        processed_image = self._process_image(image)
        obs_state = {
            "x": np.array([state["x"]], dtype=np.float32),
            "y": np.array([state["y"]], dtype=np.float32),
            "z": np.array([state["z"]], dtype=np.float32),
            "yaw": np.array([state["yaw"]], dtype=np.float32),
            "pitch": np.array([state["pitch"]], dtype=np.float32),
            "health": np.array([state["health"]], dtype=np.float32),
            "hunger": np.array([state["hunger"]], dtype=np.float32),
            "alive": int(state["alive"]),
            "inventory": {
                str(i): np.array([state["inventory"].get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
        return observation

    def _build_observation_final(self, image, state):
        """Final method to build observation."""
        processed_image = self._process_image(image)
        obs_state = {
            "x": np.array([state.get("x", 0.0)], dtype=np.float32),
            "y": np.array([state.get("y", 0.0)], dtype=np.float32),
            "z": np.array([state.get("z", 0.0)], dtype=np.float32),
            "yaw": np.array([state.get("yaw", 0.0)], dtype=np.float32),
            "pitch": np.array([state.get("pitch", 0.0)], dtype=np.float32),
            "health": np.array([state.get("health", 20.0)], dtype=np.float32),
            "hunger": np.array([state.get("hunger", 20.0)], dtype=np.float32),
            "alive": int(state.get("alive", True)),
            "inventory": {
                str(i): np.array([state.get("inventory", {}).get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
        return observation

    def _build_observation_new(self, image, state):
        """Newest method to build observation."""
        processed_image = self._process_image(image)
        obs_state = {
            "x": np.array([state.get("x", 0.0)], dtype=np.float32),
            "y": np.array([state.get("y", 0.0)], dtype=np.float32),
            "z": np.array([state.get("z", 0.0)], dtype=np.float32),
            "yaw": np.array([state.get("yaw", 0.0)], dtype=np.float32),
            "pitch": np.array([state.get("pitch", 0.0)], dtype=np.float32),
            "health": np.array([state.get("health", 20.0)], dtype=np.float32),
            "hunger": np.array([state.get("hunger", 20.0)], dtype=np.float32),
            "alive": int(state.get("alive", True)),
            "inventory": {
                str(i): np.array([state.get("inventory", {}).get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
        return observation

    def _build_observation(self, image, state):
        """Construct the observation dictionary."""
        processed_image = self._process_image(image)
        # Normalize and format state
        obs_state = {
            "x": np.array([state.get("x", 0.0)], dtype=np.float32),
            "y": np.array([state.get("y", 0.0)], dtype=np.float32),
            "z": np.array([state.get("z", 0.0)], dtype=np.float32),
            "yaw": np.array([state.get("yaw", 0.0)], dtype=np.float32),
            "pitch": np.array([state.get("pitch", 0.0)], dtype=np.float32),
            "health": np.array([state.get("health", 20.0)], dtype=np.float32),
            "hunger": np.array([state.get("hunger", 20.0)], dtype=np.float32),
            "alive": int(state.get("alive", True)),
            "inventory": {
                str(i): np.array([state.get("inventory", {}).get(str(i), 0)], dtype=np.int32)
                for i in range(9)
            }
        }
        observation = {
            "image": processed_image,
            "state": obs_state
        }
        return observation

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
        truncated = False  # Define truncation conditions if any

        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.

        Args:
            seed (int, optional): The seed for the environment's random number generator.
            options (dict, optional): Additional information for resetting the environment.

        Returns:
            observation (dict): The initial observation.
            info (dict): Diagnostic information.
        """
        if self.DEBUG:
            self.logger.debug(f"Resetting environment at {datetime.now()}")

        # Optionally send a reset action to the mod
        # Here, we choose "reset 1" to teleport to initial coordinates
        self._send_action(17)  # "reset 0"
        self._send_action(18)  # "reset 1"
        self._send_action(19)  # "reset 2"

        # Receive the initial state
        state = self._receive_state()

        # Capture initial screenshot
        image = self._capture_screenshot()

        # Build initial observation
        observation = self._build_observation(image, state)

        info = {}

        return observation, info

    def render(self, mode="human"):
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

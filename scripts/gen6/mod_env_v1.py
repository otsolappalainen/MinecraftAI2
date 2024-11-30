# File: minecraft_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import websockets
import threading
import json
import time
import cv2
from PIL import Image
import mss
import pygetwindow as gw
import fnmatch
import os
import ctypes
import logging

class MinecraftEnv(gym.Env):
    """
    Custom Environment that interfaces with Minecraft via WebSocket.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MinecraftEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects

        # Action space: Discrete actions as per your action list
        self.ACTION_MAPPING = {
            0: "move_forward",
            1: "move_backward",
            2: "move_left",
            3: "move_right",
            4: "jump_walk_forward",
            5: "jump",
            6: "sneak",
            7: "look_left",
            8: "look_right",
            9: "look_up",
            10: "look_down",
            11: "turn_left",
            12: "turn_right",
            13: "attack",
            14: "use",
            15: "next_item",
            16: "previous_item",
            17: "no_op"
        }

        self.action_space = spaces.Discrete(len(self.ACTION_MAPPING))

        # Observation space: Image (3, 224, 224), and other data
        # Define the observation space accordingly
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32),
            'other': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        })

        # Initialize WebSocket connection parameters
        self.uri = "ws://localhost:8080"
        self.websocket = None
        self.loop = None
        self.connection_thread = None
        self.connected = False

        # Initialize screenshot parameters
        self.sct = mss.mss()
        self.minecraft_bounds = self.find_minecraft_window()

        # Initialize state variables
        self.state = None

        # Initialize action and state locks
        self.action_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # Start WebSocket connection in a separate thread
        self.start_connection()

    def find_minecraft_window(self):
        """
        Finds the Minecraft window and returns its bounding box.
        If not found, raises an exception.
        """
        patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
        windows = gw.getAllTitles()
        for title in windows:
            for pattern in patterns:
                if fnmatch.fnmatch(title, pattern):
                    matched_windows = gw.getWindowsWithTitle(title)
                    if matched_windows:
                        minecraft_window = matched_windows[0]
                        if minecraft_window.isMinimized:
                            minecraft_window.restore()
                            time.sleep(0.5)  # Give time for window to restore
                        return {
                            "left": minecraft_window.left,
                            "top": minecraft_window.top,
                            "width": minecraft_window.width,
                            "height": minecraft_window.height,
                        }
        # If no matching window is found
        raise Exception("Minecraft window not found. Ensure the game is running and visible.")

    def start_connection(self):
        """
        Starts the WebSocket connection in a separate thread.
        """
        self.loop = asyncio.new_event_loop()
        self.connection_thread = threading.Thread(target=self.run_loop, args=(self.loop,), daemon=True)
        self.connection_thread.start()

    def run_loop(self, loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.connect())
        except Exception as e:
            logging.error(f"Error in run_loop: {e}")
        finally:
            self.connected = False

    async def connect(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                self.connected = True
                # Keep the connection open
                async for message in websocket:
                    # Update the state
                    with self.state_lock:
                        self.state = json.loads(message)
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            self.connected = False

    def send_action(self, action_name):
        """
        Sends an action to the Minecraft mod via WebSocket.
        """
        if self.connected and self.websocket is not None:
            message = {'action': action_name}
            asyncio.run_coroutine_threadsafe(self.websocket.send(json.dumps(message)), self.loop)
        else:
            logging.warning("Not connected to WebSocket server.")

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        """
        super().reset(seed=seed)

        # Send reset command to the mod
        self.send_action("reset 0")  # You can adjust the reset type as needed
        time.sleep(0.5)  # Wait for the game to reset

        # Get initial observation
        observation = self.get_observation()
        info = {}  # Additional info if needed
        return observation, info

    def step(self, action):
        """
        Executes an action in the environment.
        """
        action_name = self.ACTION_MAPPING.get(action, "no_op")
        self.send_action(action_name)

        # Wait for the state to update
        time.sleep(0.1)  # Adjust as needed

        # Get observation
        observation = self.get_observation()

        # For now, set reward, done, info to placeholders
        reward = 0  # You can define a reward function based on your goals
        done = False  # Define termination conditions
        truncated = False  # Define truncation conditions
        info = {}

        return observation, reward, done, truncated, info

    def get_observation(self):
        """
        Captures the game screen and combines it with other state information.
        """
        # Capture screenshot
        try:
            sct_img = self.sct.grab(self.minecraft_bounds)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        except Exception as e:
            logging.error(f"Error capturing Minecraft window: {e}")
            # Create a black image as fallback
            img = Image.new('RGB', (224, 224), color='black')

        img = img.resize((224, 224))
        image_array = np.array(img).transpose((2, 0, 1))  # Shape: (3, 224, 224)
        image_array = image_array.astype(np.float32) / 255.0  # Normalize image

        # Get state
        with self.state_lock:
            state = self.state

        # If state is not available, create a default state
        if state is None:
            state = {
                'x': 0,
                'y': 0,
                'z': 0,
                'yaw': 0,
                'pitch': 0,
                'health': 20,
                'hunger': 20,
                'alive': True
            }

        # Extract and normalize state variables
        x = state.get('x', 0)
        y_coord = state.get('y', 0)
        z = state.get('z', 0)
        yaw = state.get('yaw', 0)
        health = state.get('health', 20)
        hunger = state.get('hunger', 20)
        alive = float(state.get('alive', True))

        normalized_x = x / 100.0
        normalized_y = y_coord / 100.0
        normalized_z = z / 100.0
        sin_yaw = np.sin(np.deg2rad(yaw))
        cos_yaw = np.cos(np.deg2rad(yaw))
        normalized_health = health / 20.0
        normalized_hunger = hunger / 20.0

        other = np.array([
            normalized_x,
            normalized_z,
            normalized_y,
            sin_yaw,
            cos_yaw,
            normalized_health,
            normalized_hunger,
            alive
        ], dtype=np.float32)

        observation = {
            'image': image_array,
            'other': other
        }

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        """
        Closes the environment and the WebSocket connection.
        """
        self.connected = False
        if self.websocket is not None:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.connection_thread is not None:
            self.connection_thread.join()
        self.sct.close()

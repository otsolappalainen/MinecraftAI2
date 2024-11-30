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
from pynput import keyboard
from queue import Queue

class MinecraftEnv(gym.Env):
    """
    Custom Environment that interfaces with Minecraft via WebSocket.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, task=None):
        super(MinecraftEnv, self).__init__()

        # Define action and observation space
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
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32),
            'other': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            'task': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
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

        # Initialize task
        if task is None:
            self.possible_tasks = [
                np.array([0, 1] + [0]*18, dtype=np.float32),  # Example task
            ]
            self.task = self.possible_tasks[0]
        else:
            self.task = np.array(task, dtype=np.float32)

        # Initialize mouse listener (if needed)
        self.mouse_action_queue = Queue()
        self.mouse_capture = None  # You can initialize this if needed
        # self.mouse_capture.start()

        # Initialize keyboard listener (if needed)
        self.keyboard_listener = None  # You can initialize this if needed
        # self.keyboard_listener.start()

        # Data directory setup for saving collected data
        self.DATA_DIR = 'expert_data'
        os.makedirs(self.DATA_DIR, exist_ok=True)

        # Create a unique session directory
        self.SESSION_ID = time.strftime('%Y%m%d_%H%M%S')
        self.SESSION_DIR = os.path.join(self.DATA_DIR, f'session_{self.SESSION_ID}')
        os.makedirs(self.SESSION_DIR, exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def step(self, action):
        """
        Executes one step of the environment.
        """
        # Process action
        action_name = self.ACTION_MAPPING[action]
        self.send_action(action_name)

        # Capture the current state (image + task data)
        screenshot = self.capture_screenshot()

        # Collect state information (e.g., task)
        state_data = {
            'image': screenshot,
            'other': np.zeros(8),  # Replace with actual state data if needed
            'task': self.task
        }

        # Get reward (for now, dummy reward)
        reward = 0.0
        done = False  # Modify based on your stopping condition

        return state_data, reward, done, {}

    def capture_screenshot(self):
        """
        Captures a screenshot from the Minecraft window.
        """
        screenshot = self.sct.grab(self.minecraft_bounds)
        image = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
        image = image.resize((224, 224))  # Resize to 224x224 if needed
        return np.array(image) / 255.0  # Normalize the image

    def reset(self):
        """
        Resets the environment to an initial state by sending a no_op action to the WebSocket.
        """
        self.send_action("no_op")  # Sending the no_op action to get the starting state
        time.sleep(1)  # Sleep to allow the state to be updated
        return {
            'image': np.zeros((224, 224, 3)),  # Replace with actual image after capturing
            'other': np.zeros(8),  # Replace with actual state data if available
            'task': self.task
        }

    def render(self, mode='human'):
        """
        Renders the environment (if necessary).
        """
        pass  # Rendering logic if necessary

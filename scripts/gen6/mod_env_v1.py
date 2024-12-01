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

        # Hardcoded save_screenshots flag
        self.save_screenshots = False  # Set to True to save screenshots

        # Ensure the screenshot save directory exists
        if self.save_screenshots:
            self.screenshot_dir = "env_screenshots"
            os.makedirs(self.screenshot_dir, exist_ok=True)

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
                logging.info("Connected to WebSocket server.")
                
                # Keep the connection open
                while self.connected:
                    try:
                        message = await websocket.recv()
                        with self.state_lock:
                            self.state = json.loads(message)
                            logging.debug(f"Received state: {self.state}")
                    except websockets.ConnectionClosed:
                        logging.warning("WebSocket connection closed.")
                        self.connected = False
                        break
                    except Exception as e:
                        logging.error(f"Error receiving state from mod: {e}")
                        await asyncio.sleep(0.1)  # Retry receiving state after a short delay
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            self.connected = False

    def capture_screenshot(self):
        """
        Captures a screenshot from the Minecraft window and saves it if required.
        """
        screenshot = self.sct.grab(self.minecraft_bounds)
        image = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
        image = image.resize((224, 224))  # Resize to 224x224 if needed
        screenshot_array = np.array(image) / 255.0  # Normalize the image

        # Save the screenshot if the flag is True
        if self.save_screenshots:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
            image.save(screenshot_path)
            logging.info(f"Saved screenshot at {screenshot_path}")

        return screenshot_array

    async def send_action(self, action_name):
        """
        Sends an action to the Minecraft mod via WebSocket.
        """
        if self.connected and self.websocket is not None:
            message = {'action': action_name}
            try:
                await self.websocket.send(json.dumps(message))
                logging.debug(f"Sent action: {action_name}")
            except Exception as e:
                logging.error(f"Error sending action to mod: {e}")
    

    async def step(self, action):
        """
        Executes one step of the environment.
        """
        action_name = self.ACTION_MAPPING[action]
        #print(f"step called: {action_name}")
        # Send action to mod
        await self.send_action(action_name)
        
        #print(f"await self.send_action(action_name) called")

        # Capture screenshot
        screenshot = self.capture_screenshot()

        #print(f"screenshot captured")

        # Wait for the state to be updated via WebSocket (with timeout)
        timeout = 5  # Maximum time to wait for state update
        start_time = time.time()

        # Async loop to wait for state from WebSocket
        while self.state is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)  # Non-blocking sleep

        # If state is not received in time, return default values
        if self.state is None:
            logging.warning(f"State not received after sending action: {action_name}")
            state_data = {
                'image': np.zeros((224, 224, 3), dtype=np.float32),  # Placeholder for image
                'other': np.zeros(8, dtype=np.float32),  # Placeholder for scalar state data
                'task': self.task  # Return current task
            }
            reward = 0.0
            done = False
            truncated = False
            info = {}
            return state_data, reward, done, truncated, info

        # Map incoming state to the expected "other" format
        other = np.array([
            self.state.get('x', 0.0),
            self.state.get('y', 0.0),
            self.state.get('z', 0.0),
            self.state.get('health', 20.0),
            self.state.get('hunger', 20.0),
            self.state.get('pitch', 0.0),
            self.state.get('yaw', 0.0),
            self.state.get('alive', 0.0)
        ], dtype=np.float32)

        task = self.state.get('task', self.task)

        # Return the state in the correct format
        state_data = {
            'image': screenshot,  # Use the captured screenshot
            'other': other,       # Use the mapped 'other' data
            'task': task
        }

        reward = 0.0  # Modify with actual reward calculation
        done = False  # Modify based on your stopping condition
        truncated = False  # Modify based on your stopping condition

        info = {}  # Add any additional info if needed
        return state_data, reward, done, truncated, info




    async def reset(self):
        """
        Resets the environment to an initial state by sending a no_op action to the WebSocket.
        Done flag is always set to False for now.
        """
        # Send the no_op action to get the starting state
        await self.send_action("no_op")

        # Wait for the state to be updated via WebSocket (with timeout)
        timeout = 2  # Maximum time to wait for state update
        start_time = time.time()

        while self.state is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)  # Non-blocking sleep

        # Capture screenshot for the image
        screenshot = self.capture_screenshot()

        # Log the state received from the WebSocket (like in the step function)
        if self.state is not None:
            logging.debug(f"Received state during reset: {self.state}")
        else:
            logging.warning("State not received during reset.")

        # Return the state data or placeholder if state is not updated
        if self.state is None:
            logging.warning("State not received during reset.")
            state_data = {
                'image': screenshot,  # Capture screenshot if state is not updated
                'other': np.zeros(8, dtype=np.float32),
                'task': self.task
            }
            return state_data, {}  # Return state_data and an empty info dictionary

        # Add screenshot to the state and return
        state_data = {
            'image': screenshot,  # Include the captured screenshot
            'other': self.state.get('other', np.zeros(8, dtype=np.float32)),  # Ensure 'other' key exists
            'task': self.state.get('task', self.task)  # Ensure 'task' key exists
        }

        return state_data, {}  # Return the state with the image and an empty info dictionary


    def render(self, mode='human'):
        """
        Render the environment (if needed).
        """
        pass  # You can implement visualization code here

    def close(self):
        """
        Close the environment and WebSocket connection.
        """
        self.connected = False
        if self.websocket is not None:
            self.websocket.close()

    def __del__(self):
        """
        Cleanup before deleting the environment object.
        """
        self.close()

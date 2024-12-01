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
        max_blocks = 5
        block_features = 4  # blocktype, x, y, z per block
        basic_features = 8  # x, y, z, health, hunger, pitch, yaw, alive

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32),
            'other': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(basic_features + max_blocks * block_features,), 
                dtype=np.float32
            ),
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
                
                while self.connected:
                    try:
                        message = await websocket.recv()
                        with self.state_lock:
                            self.state = json.loads(message)
                            logging.debug(f"Received state: {self.state}")
                    except websockets.ConnectionClosed:
                        logging.warning("WebSocket connection closed.")
                        break
                    except Exception as e:
                        logging.error(f"Error receiving state: {e}")
                        await asyncio.sleep(0.1)
        
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
        finally:
            self.connected = False
            self.websocket = None

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
        await self.send_action(action_name)
        screenshot = self.capture_screenshot()

        # Wait for state update with timeout
        timeout = 5
        start_time = time.time()
        while self.state is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.05)

        # Return default state if no state received
        if self.state is None:
            logging.warning(f"State not received after sending action: {action_name}")
            return self._get_default_state(), 0.0, False, False, {}

        # Initialize arrays for basic state and broken blocks
        max_blocks = 5
        block_features = 4  # blocktype, x, y, z per block
        basic_features = 8  # x, y, z, health, hunger, pitch, yaw, alive

        # Process basic state data
        basic_state = np.array([
            self.state.get('x', 0.0),
            self.state.get('y', 0.0),
            self.state.get('z', 0.0),
            self.state.get('health', 20.0),
            self.state.get('hunger', 20.0),
            self.state.get('pitch', 0.0),
            self.state.get('yaw', 0.0),
            self.state.get('alive', 0.0)
        ], dtype=np.float32)

        # Process broken blocks data - always ensure 20 values (5 blocks × 4 features)
        broken_blocks_array = np.zeros(max_blocks * block_features, dtype=np.float32)
        broken_blocks = self.state.get('broken_blocks', [])
        
        for i, block in enumerate(broken_blocks[:max_blocks]):
            idx = i * block_features
            broken_blocks_array[idx] = float(block.get('blocktype', 0))
            broken_blocks_array[idx + 1] = float(block.get('blockx', 0))
            broken_blocks_array[idx + 2] = float(block.get('blocky', 0))
            broken_blocks_array[idx + 3] = float(block.get('blockz', 0))

        # Combine basic state and broken blocks data
        other = np.concatenate([basic_state, broken_blocks_array])

        # Return the state in the correct format
        state_data = {
            'image': screenshot,
            'other': other,  # Will always be 28 values (8 basic + 20 block features)
            'task': self.state.get('task', self.task)
        }

        reward = 0.0
        done = False
        truncated = False
        info = {'broken_blocks': len(broken_blocks)}

        return state_data, reward, done, truncated, info

    async def reset(self):
        """Reset the environment"""
        if not self.connected or self.websocket is None:
            logging.error("WebSocket not connected")
            return self._get_default_state(), {}
        
        try:
            with self.state_lock:
                self.state = None
                
            await self.websocket.send('{"action": "reset 2"}')
            
            timeout = 2.0
            start_time = time.time()
            
            while self.state is None and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.1)
            
            if self.state is None:
                logging.warning("Reset: No state received")
                return self._get_default_state(), {}
                
            screenshot = self.capture_screenshot()
            
            # Create zeros array for broken blocks (20 values)
            broken_blocks_array = np.zeros(20, dtype=np.float32)
            
            # Basic state with broken blocks array
            other = np.concatenate([
                np.array([
                    self.state.get('x', 0.0),
                    self.state.get('y', 0.0), 
                    self.state.get('z', 0.0),
                    self.state.get('health', 20.0),
                    self.state.get('hunger', 20.0),
                    self.state.get('pitch', 0.0),
                    self.state.get('yaw', 0.0),
                    self.state.get('alive', 1.0)
                ], dtype=np.float32),
                broken_blocks_array  # Add 20 zeros for broken blocks
            ])
            
            state_data = {
                'image': screenshot,
                'other': other,  # Will be 28 values (8 basic + 20 zeros)
                'task': self.task
            }
            
            return state_data, {}
            
        except Exception as e:
            logging.error(f"Reset error: {e}")
            return self._get_default_state(), {}

    def _get_default_state(self):
        """Return default state when real state cannot be obtained"""
        return {
            'image': np.zeros((3, 224, 224), dtype=np.float32),
            'other': np.zeros(8, dtype=np.float32),
            'task': self.task
        }

    def render(self, mode='human'):
        """
        Render the environment (if needed).
        """
        pass  # You can implement visualization code here

    def close(self):
        """Close the environment"""
        if self.connected and self.websocket:
            self.connected = False
            # Schedule websocket.close() in the event loop
            future = asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                logging.error(f"Error closing websocket: {e}")
            self.websocket = None

        # Stop the event loop safely
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        # Join the connection thread
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=1)

    def __del__(self):
        """
        Cleanup before deleting the environment object.
        """
        self.close()

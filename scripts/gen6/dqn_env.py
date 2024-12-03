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
import logging
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
        self.minecraft_bounds = self.find_minecraft_window()

        # Initialize action and state queues
        self.state_queue = asyncio.Queue()

        # Start WebSocket connection in a separate thread
        self.start_connection()

        # Initialize task with height targets
        self.default_task = np.array([0, 0, -63, -62] + [0]*16, dtype=np.float32)
        self.task = self.default_task if task is None else np.array(task, dtype=np.float32)

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize step counters and parameters
        self.steps = 0
        self.max_episode_steps = 400
        self.target_height_min = -63
        self.target_height_max = -60
        self.block_break_reward = 3.0
        self.height_penalty = -0.5
        self.step_penalty = -0.1

        # Normalization ranges
        self.coord_range = (-10000, 10000)  # Typical Minecraft coordinate range
        self.height_range = (-256, 256)  # Minecraft height range
        self.angle_range = (-180, 180)  # Pitch/yaw range
        self.health_range = (0, 20)  # Health range
        self.hunger_range = (0, 20)  # Hunger range

        # Cumulative reward and episode count
        self.cumulative_reward = 0.0
        self.episode_count = 0

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
                        state = json.loads(message)
                        await self.state_queue.put(state)
                        logging.debug(f"Received state: {state}")
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

    async def send_action(self, action_name):
        """Async implementation of send_action"""
        if self.connected and self.websocket is not None:
            message = {'action': action_name}
            try:
                await self.websocket.send(json.dumps(message))
                logging.debug(f"Sent action: {action_name}")
            except Exception as e:
                logging.error(f"Error sending action to mod: {e}")

    def normalize_value(self, value, range_min, range_max):
        """Normalize a value to [-1, 1] range"""
        return 2 * (value - range_min) / (range_max - range_min) - 1

    def normalize_basic_state(self, state_dict):
        """Normalize basic state values to [-1, 1] range"""
        return np.array([
            self.normalize_value(state_dict.get('x', 0), *self.coord_range),
            self.normalize_value(state_dict.get('y', 0), *self.height_range),
            self.normalize_value(state_dict.get('z', 0), *self.coord_range),
            self.normalize_value(state_dict.get('health', 20), *self.health_range),
            self.normalize_value(state_dict.get('hunger', 20), *self.hunger_range),
            self.normalize_value(state_dict.get('pitch', 0), *self.angle_range),
            self.normalize_value(state_dict.get('yaw', 0), *self.angle_range),
            state_dict.get('alive', 0)  # Already binary
        ], dtype=np.float32)

    def normalize_block_data(self, block):
        """Normalize block coordinates to [-1, 1] range"""
        return [
            float(block.get('blocktype', 0)) / 1000,  # Normalize block type
            self.normalize_value(float(block.get('blockx', 0)), *self.coord_range),
            self.normalize_value(float(block.get('blocky', 0)), *self.height_range),
            self.normalize_value(float(block.get('blockz', 0)), *self.coord_range)
        ]

    def step(self, action):
        """Gym synchronous step method"""
        start_time = time.time()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._step(action))
        elapsed_time = time.time() - start_time
        logging.debug(f"Step time: {elapsed_time:.4f} seconds")
        return result

    async def _step(self, action):
        self.steps += 1
        
        action_name = self.ACTION_MAPPING[action]
        
        # Clear the state queue before sending action
        while not self.state_queue.empty():
            self.state_queue.get_nowait()

        # Schedule sending the action in the event loop
        send_action_start_time = time.time()
        await self.send_action(action_name)
        send_action_elapsed_time = time.time() - send_action_start_time
        logging.debug(f"Send action time: {send_action_elapsed_time:.4f} seconds")

        # Capture screenshot asynchronously
        screenshot_start_time = time.time()
        screenshot = await asyncio.to_thread(self.capture_screenshot)
        screenshot_elapsed_time = time.time() - screenshot_start_time
        logging.debug(f"Screenshot capture time: {screenshot_elapsed_time:.4f} seconds")

        # Wait for state update with timeout
        timeout = 0.15
        try:
            state_start_time = time.time()
            logging.debug("Waiting for state update...")
            state = await asyncio.wait_for(self.state_queue.get(), timeout)
            state_elapsed_time = time.time() - state_start_time
            logging.debug(f"State wait time: {state_elapsed_time:.4f} seconds")
        except asyncio.TimeoutError:
            state = None
            logging.warning(f"State not received after sending action: {action_name}")

        # Calculate reward
        reward_calc_start_time = time.time()
        reward = self.step_penalty  # Default penalty per step
        done = False
        broken_blocks = []

        if state is not None:
            # Get player height
            player_y = state.get('y', 0.0)
            
            # Height penalty if outside target range
            if not (self.target_height_min <= player_y <= self.target_height_max):
                reward += self.height_penalty
            
            # Reward for breaking blocks at correct height
            broken_blocks = state.get('broken_blocks', [])
            for block in broken_blocks:
                block_y = float(block.get('blocky', 0))
                if block_y in [-63, -62]:
                    reward += self.block_break_reward
        else:
            logging.warning(f"No state received after action: {action_name}")
            
        # Check if episode should end
        done = self.steps >= self.max_episode_steps
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # Print progress every 50 steps
        if self.steps % 50 == 0:
            print(f"Episode {self.episode_count} - Step {self.steps}/{self.max_episode_steps}, "
                  f"Cumulative reward: {self.cumulative_reward:.2f}")
        
        # If episode is done, print final stats
        if done:
            print(f"\nEpisode {self.episode_count} finished:")
            print(f"Total steps: {self.steps}")
            print(f"Final cumulative reward: {self.cumulative_reward:.2f}\n")
            self.episode_count += 1

        reward_calc_elapsed_time = time.time() - reward_calc_start_time
        logging.debug(f"Reward calculation time: {reward_calc_elapsed_time:.4f} seconds")

        # Get observation state
        state_prep_start_time = time.time()
        if state is None:
            state_data = self._get_default_state()
        else:
            max_blocks = 5
            block_features = 4
            
            basic_state = self.normalize_basic_state(state)
            
            # Normalize block data
            broken_blocks_array = np.zeros(max_blocks * block_features, dtype=np.float32)
            for i, block in enumerate(broken_blocks[:max_blocks]):
                idx = i * block_features
                normalized_block = self.normalize_block_data(block)
                broken_blocks_array[idx:idx + block_features] = normalized_block

            other = np.concatenate([basic_state, broken_blocks_array])
            
            state_data = {
                'image': screenshot,  # Already normalized in capture_screenshot
                'other': other,
                'task': self.task  # Task remains the same
            }

        state_prep_elapsed_time = time.time() - state_prep_start_time
        logging.debug(f"State preparation time: {state_prep_elapsed_time:.4f} seconds")

        info = {
            'broken_blocks': len(broken_blocks),
            'reward_breakdown': {
                'step_penalty': self.step_penalty,
                'height_penalty': self.height_penalty if state and not (self.target_height_min <= state.get('y', 0.0) <= self.target_height_max) else 0,
                'block_breaks': sum(1 for block in broken_blocks if float(block.get('blocky', 0)) in [-63, -62]) * self.block_break_reward
            }
        }

        return state_data, reward, done, False, info

    def reset(self, *, seed=None, options=None):
        """Gym synchronous reset method"""
        start_time = time.time()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._reset(seed=seed, options=options))
        elapsed_time = time.time() - start_time
        logging.debug(f"Reset time: {elapsed_time:.4f} seconds")
        return result

    async def _reset(self, *, seed=None, options=None):
        self.steps = 0  # Reset step counter
        self.cumulative_reward = 0.0
        
        if seed is not None:
            np.random.seed(seed)
        
        if not self.connected or self.websocket is None:
            logging.error("WebSocket not connected")
            state_data = self._get_default_state()
            info = {}
            return state_data, info

        try:
            # Clear the state queue
            while not self.state_queue.empty():
                self.state_queue.get_nowait()
                
            # Send reset action
            await self.send_action("reset 2")
            
            timeout = 2.0
            try:
                state_start_time = time.time()
                state = await asyncio.wait_for(self.state_queue.get(), timeout)
                state_elapsed_time = time.time() - state_start_time
                logging.debug(f"State wait time (reset): {state_elapsed_time:.4f} seconds")
            except asyncio.TimeoutError:
                state = None
                logging.warning("Reset: No state received")
                state_data = self._get_default_state()
                info = {}
                return state_data, info

            # Capture screenshot asynchronously
            screenshot_start_time = time.time()
            screenshot = await asyncio.to_thread(self.capture_screenshot)
            screenshot_elapsed_time = time.time() - screenshot_start_time
            logging.debug(f"Screenshot capture time (reset): {screenshot_elapsed_time:.4f} seconds")
            
            # Prepare state data
            max_blocks = 5
            block_features = 4
            broken_blocks_array = np.zeros(max_blocks * block_features, dtype=np.float32)
            basic_state = self.normalize_basic_state(state)
            other = np.concatenate([basic_state, broken_blocks_array])
            
            state_data = {
                'image': screenshot,
                'other': other,
                'task': self.task
            }
            
            return state_data, {}  # Return state and empty info
            
        except Exception as e:
            logging.error(f"Reset error: {e}")
            state_data = self._get_default_state()
            info = {}
            return state_data, info

    def capture_screenshot(self):
        """
        Captures a screenshot from the Minecraft window and saves it if required.
        Returns image in CHW format (PyTorch style).
        """
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(self.minecraft_bounds)
                image = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
                image = image.resize((224, 224))
                
                # Convert to numpy array and normalize
                screenshot_array = np.array(image) / 255.0  # Shape: (H, W, C)
                
                # Transpose from HWC to CHW format
                screenshot_array = screenshot_array.transpose(2, 0, 1)  # Shape: (C, H, W)
        
                # Save the screenshot if the flag is True
                if self.save_screenshots:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
                    image.save(screenshot_path)
                    logging.info(f"Saved screenshot at {screenshot_path}")
        
                return screenshot_array.astype(np.float32)  # Ensure float32 type
        except Exception as e:
            logging.error(f"Error in capture_screenshot: {e}")
            # Return a default image to prevent crashing
            return np.zeros((3, 224, 224), dtype=np.float32)

    def _get_default_state(self):
        """Return default state when real state cannot be obtained"""
        basic_features = 8
        max_blocks = 5
        block_features = 4
        total_features = basic_features + (max_blocks * block_features)
        
        return {
            'image': np.zeros((3, 224, 224), dtype=np.float32),
            'other': np.zeros(total_features, dtype=np.float32),
            'task': self.default_task
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

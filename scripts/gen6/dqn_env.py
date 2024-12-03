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

def run_coroutine(coroutine):
    """Helper function to run coroutines from sync code"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

class StepTimer:
    """Manages timing for environment steps to maintain consistent rate"""
    def __init__(self, target_step_time=0.1):
        self.target_step_time = target_step_time
        self.last_step_time = time.time()
        self.start_time = self.last_step_time
        self.step_count = 0
        
    async def wait_for_next_step(self):
        """Calculate and wait required time until next step"""
        current_time = time.time()
        elapsed = current_time - self.last_step_time
        
        if (elapsed < self.target_step_time):
            await asyncio.sleep(self.target_step_time - elapsed)
            
        self.last_step_time = time.time()
        self.step_count += 1
        
    def get_stats(self):
        """Get timing statistics"""
        return {
            'avg_step_time': (time.time() - self.start_time) / max(1, self.step_count),
            'total_steps': self.step_count,
            'total_time': time.time() - self.start_time
        }

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

        # Initialize task with height targets
        self.default_task = np.array([0, 0, 132, 133] + [0]*16, dtype=np.float32)
        self.task = self.default_task if task is None else np.array(task, dtype=np.float32)

        # Initialize mouse listener (if needed)
        self.mouse_action_queue = Queue()
        self.mouse_capture = None  # You can initialize this if needed
        # self.mouse_capture.start()

        # Initialize keyboard listener (if needed)
        self.keyboard_listener = None  # You can initialize this if needed
        # self.keyboard_listener.start()

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Add after existing init code:
        self.steps = 0
        self.max_episode_steps = 300
        self.target_height_min = 131
        self.target_height_max = 133
        self.block_break_reward = 3.0
        self.height_penalty = -0.2
        self.step_penalty = -0.1

        # Add after other init variables:
        
        # Normalization ranges
        self.coord_range = (-10000, 10000)  # Typical Minecraft coordinate range
        self.height_range = (-256, 256)  # Minecraft height range
        self.angle_range = (-180, 180)  # Pitch/yaw range
        self.health_range = (0, 20)  # Health range
        self.hunger_range = (0, 20)  # Hunger range

        # Add these lines:
        self.cumulative_reward = 0.0
        self.episode_count = 0

        # Add after other init code:
        self.timer = StepTimer(target_step_time=0.1)

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
        Returns image in CHW format (PyTorch style).
        """
        screenshot = self.sct.grab(self.minecraft_bounds)
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

    def send_action(self, action_name):
        """Synchronous wrapper for async send_action"""
        return run_coroutine(self._send_action(action_name))

    async def _send_action(self, action_name):
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
        """Synchronous wrapper for async step"""
        result = run_coroutine(self._step(action))
        return result

    def reset(self, *, seed=None, options=None):
        """Synchronous wrapper for async reset"""
        obs, info = run_coroutine(self._reset(seed=seed, options=options))
        return obs, info

    async def _step(self, action):
        """Async implementation of step"""
        self.steps += 1
        
        action_name = self.ACTION_MAPPING[action]
        await self._send_action(action_name)
        screenshot = self.capture_screenshot()

        # Wait for state update with timeout
        timeout = 1
        start_time = time.time()
        while self.state is None and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.02)

        # Calculate reward
        reward = self.step_penalty  # Default penalty per step
        done = False
        
        if self.state is not None:
            # Get player height
            player_y = self.state.get('y', 0.0)
            
            # Height penalty if outside target range
            if not (self.target_height_min <= player_y <= self.target_height_max):
                reward += self.height_penalty
            
            # Reward for breaking blocks at correct height
            broken_blocks = self.state.get('broken_blocks', [])
            for block in broken_blocks:
                block_y = float(block.get('blocky', 0))
                if block_y in [132, 133]:
                    reward += self.block_break_reward
        else:
            logging.warning(f"State not received after sending action: {action_name}")
            
        # Check if episode should end
        done = self.steps >= self.max_episode_steps
        
        # After reward calculation:
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

        # Get observation state
        if self.state is None:
            state_data = self._get_default_state()
        else:
            max_blocks = 5
            block_features = 4
            
            basic_state = self.normalize_basic_state(self.state)
            
            # Normalize block data
            broken_blocks_array = np.zeros(max_blocks * block_features, dtype=np.float32)
            broken_blocks = self.state.get('broken_blocks', [])
            for i, block in enumerate(broken_blocks[:max_blocks]):
                idx = i * block_features
                normalized_block = self.normalize_block_data(block)
                broken_blocks_array[idx:idx + block_features] = normalized_block

            other = np.concatenate([basic_state, broken_blocks_array])
            
            state_data = {
                'image': screenshot,  # Already normalized in capture_screenshot
                'other': other,
                'task': self.state.get('task', self.task)  # Already normalized
            }

        # Before return, wait for next step timing
        await self.timer.wait_for_next_step()
        
        if self.steps % 50 == 0:
            stats = self.timer.get_stats()
            print(f"Timing stats - Avg step: {stats['avg_step_time']:.3f}s, "
                  f"Total time: {stats['total_time']:.1f}s")

        info = {
            'broken_blocks': len(broken_blocks) if self.state else 0,
            'reward_breakdown': {
                'step_penalty': self.step_penalty,
                'height_penalty': self.height_penalty if not (self.target_height_min <= self.state.get('y', 0.0) <= self.target_height_max) else 0,
                'block_breaks': sum(1 for block in (broken_blocks if self.state else []) 
                                  if float(block.get('blocky', 0)) in [132, 133]) * self.block_break_reward
            }
        }

        return state_data, reward, done, False, info

    async def _reset(self, *, seed=None, options=None):
        """Async implementation of reset"""
        self.steps = 0  # Reset step counter
        
        if seed is not None:
            np.random.seed(seed)
        
        if not self.connected or self.websocket is None:
            logging.error("WebSocket not connected")
            state_data = self._get_default_state()
            info = {}
            return state_data, info  # Return tuple instead of dict
        
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
                state_data = self._get_default_state()
                info = {}
                return state_data, info  # Return tuple instead of dict
                
            screenshot = self.capture_screenshot()
            broken_blocks_array = np.zeros(20, dtype=np.float32)
            
            basic_state = self.normalize_basic_state(self.state)
            other = np.concatenate([basic_state, broken_blocks_array])
            
            state_data = {
                'image': screenshot,
                'other': other,
                'task': self.task
            }
            
            # Reset cumulative reward at start of episode
            self.cumulative_reward = 0.0
            
            return state_data, {}  # Return tuple instead of dict
            
        except Exception as e:
            logging.error(f"Reset error: {e}")
            state_data = self._get_default_state()
            info = {}
            return state_data, info  # Return tuple instead of dict

    def _get_default_state(self):
        """Return default state when real state cannot be obtained"""
        basic_features = 8
        max_blocks = 5
        block_features = 4
        total_features = basic_features + (max_blocks * block_features)
        
        return {
            'image': np.zeros((3, 224, 224), dtype=np.float32),
            'other': np.zeros(total_features, dtype=np.float32),  # 28 total features
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

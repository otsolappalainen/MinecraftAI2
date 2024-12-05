import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
import websockets
import threading
import json
import time
import mss
import cv2
import pygetwindow as gw
import fnmatch
import os
import logging

class MinecraftEnv(gym.Env):
    """
    Custom Environment that interfaces with Minecraft via WebSocket.
    Handles a single client per environment instance.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, uri="ws://localhost:8080", task=None, window_bounds=None):
        super(MinecraftEnv, self).__init__()

        # Set up logging
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        # Normalization ranges
        self.coord_range = (-10000, 10000)
        self.height_range = (-256, 256)
        self.angle_range = (-180, 180)
        self.health_range = (0, 20)
        self.hunger_range = (0, 20)

        # Single URI
        self.uri = uri

        # Screenshot settings
        self.save_screenshots = False  # Set to True to save screenshots
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

        # Observation space
        tasks_dim = 10
        blocks_dim = 4  # 4 features per block
        hand_dim = 5
        target_block_dim = 1
        flattened_matrix_dim = 729

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(3, 120, 120), dtype=np.float32),
            'tasks': spaces.Box(low=-np.inf, high=np.inf, shape=(tasks_dim,), dtype=np.float32),
            'blocks': spaces.Box(low=-np.inf, high=np.inf, shape=(blocks_dim,), dtype=np.float32),
            'hand': spaces.Box(low=0, high=1, shape=(hand_dim,), dtype=np.float32),
            'target_block': spaces.Box(low=0, high=1, shape=(target_block_dim,), dtype=np.float32),
            'flattened_matrix': spaces.Box(low=-1, high=1, shape=(flattened_matrix_dim,), dtype=np.float32)
        })

        # Initialize WebSocket connection parameters
        self.websocket = None
        self.loop = None
        self.connection_thread = None
        self.connected = False

        # Initialize screenshot parameters
        self.minecraft_bounds = window_bounds if window_bounds else self.find_minecraft_window()

        # Initialize action and state queues
        self.state_queue = asyncio.Queue()

        # Start WebSocket connection in a separate thread
        self.start_connection()

        # Initialize task with height targets
        self.default_task = self.normalize_task({})
        self.task = self.default_task if task is None else np.array(task, dtype=np.float32)

        # Initialize step counters and parameters
        self.steps = 0
        self.max_episode_steps = 100
        self.target_height_min = -63
        self.target_height_max = -60
        self.block_break_reward = 3.0
        self.height_penalty = -0.2
        self.step_penalty = -0.1

        # Cumulative reward and episode count
        self.cumulative_rewards = 0.0
        self.episode_counts = 0

    def find_minecraft_window(self):
        """
        Finds the Minecraft window matching the patterns and returns its bounds.
        """
        patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
        all_windows = gw.getAllTitles()
        matched_windows = []
        seen_handles = set()  # Track windows by handle instead of title

        for title in all_windows:
            for pattern in patterns:
                if fnmatch.fnmatch(title, pattern):
                    windows = gw.getWindowsWithTitle(title)
                    for window in windows:
                        # Use window handle as unique identifier
                        if window._hWnd not in seen_handles:
                            matched_windows.append(window)
                            seen_handles.add(window._hWnd)

        if len(matched_windows) < 1:
            raise Exception(f"No Minecraft windows found. Expected at least 1, Found: {len(matched_windows)}")

        # Sort windows by their top (y) coordinate
        sorted_windows = sorted(matched_windows, key=lambda w: w.top)

        # Assign the first sorted window to this client
        window = sorted_windows[0]
        if window.isMinimized:
            window.restore()
            time.sleep(0.1)  # Reduced sleep time

        bounds = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        }
        return bounds

    def start_connection(self):
        self.loop = asyncio.new_event_loop()
        self.connection_thread = threading.Thread(target=self.run_loop, args=(self.loop,), daemon=True)
        self.connection_thread.start()

    def run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect())

    async def connect(self):
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                self.connected = True
                logging.info(f"Connected to {self.uri}")

                while self.connected:
                    tasks = [asyncio.ensure_future(self.receive_state())]
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
        finally:
            self.connected = False
            self.websocket = None

    async def receive_state(self):
        try:
            message = await self.websocket.recv()
            state = json.loads(message)
            await self.state_queue.put(state)
        except websockets.ConnectionClosed:
            logging.warning("WebSocket connection closed.")
            self.connected = False
        except Exception as e:
            logging.error(f"Error receiving state: {e}")
            await asyncio.sleep(0.1)

    async def send_action(self, action_name):
        if self.connected and self.websocket is not None:
            message = {'action': action_name}
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending action: {e}")

    def normalize_value(self, value, range_min, range_max):
        """Normalize a value to [-1, 1] range"""
        return 2 * (value - range_min) / (range_max - range_min) - 1

    def normalize_blocks(self, broken_blocks):
        """
        Normalize broken blocks data where input is [blocktype, x, y, z]
        blocktype: 0 or 1
        If blocktype is 1, normalize coordinates:
        x: divide by 20000
        y: divide by 512
        z: divide by 20000
        """
        block_features = 4  # [blocktype, x, y, z]

        # Handle multiple broken blocks by taking the first one
        if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
            broken_blocks = broken_blocks[0]

        if isinstance(broken_blocks, list) and len(broken_blocks) == 4:
            broken_blocks = np.array(broken_blocks, dtype=np.float32)
            if broken_blocks[0] == 1:
                broken_blocks[1] /= 20000.0
                broken_blocks[2] /= 512.0
                broken_blocks[3] /= 20000.0
            else:
                broken_blocks[1:] = 0.0
            return broken_blocks
        else:
            return np.zeros(block_features, dtype=np.float32)

    def normalize_target_block(self, state_dict):
        """Normalize target block data - use direct value"""
        target_block = state_dict.get('target_block', 0)
        return np.array([float(target_block)], dtype=np.float32)

    def normalize_hand(self, state_dict):
        """Normalize hand/item data - use full array"""
        held_item = state_dict.get('held_item', [0, 0, 0, 0, 0])
        return np.array(held_item, dtype=np.float32)

    def normalize_task(self, state_dict):
        """Return constant task array with normalized height values"""
        return np.array([
            0,
            0,
            self.normalize_value(-63, *self.height_range),  # Normalize target min height
            self.normalize_value(-62, *self.height_range),  # Normalize target max height
            0,
            0,
            0,
            0,
            0,
            0
        ], dtype=np.float32)

    def flatten_surrounding_blocks(self, state_dict):
        """Flatten surrounding blocks data"""
        surrounding = state_dict.get('surrounding_blocks', [])
        flattened = np.array(surrounding, dtype=np.float32).flatten()
        if flattened.size < 729:
            padded = np.pad(flattened, (0, 729 - flattened.size), 'constant')
        else:
            padded = flattened[:729]
        return np.clip(padded, -1, 1)

    def step(self, action):
        """Handle single action"""
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")

        action = int(action)

        # Schedule the coroutine and get a Future
        future = asyncio.run_coroutine_threadsafe(
            self._async_step(action_name=self.ACTION_MAPPING[action]),
            self.loop
        )

        try:
            # Wait for the result with a timeout if necessary
            result = future.result(timeout=0.3)  # Adjust timeout as needed
            return result
        except Exception as e:
            logging.error(f"Error during step: {e}")
            raise e

    async def _async_step(self, action_name=None):
        if action_name:
            await self.send_action(action_name)

        # Capture screenshot asynchronously
        screenshot_task = asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)

        # Receive state
        try:
            state = await asyncio.wait_for(self.state_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            state = None
            logging.warning("Did not receive state in time.")

        # Wait for screenshot to complete
        screenshot = await screenshot_task

        # Process state
        reward = self.step_penalty
        done = False
        broken_blocks = []

        if state is not None:
            # Process the state as before
            player_y = state.get('y', 0.0)
            if not (self.target_height_min <= player_y <= self.target_height_max):
                reward += self.height_penalty

            broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
            if isinstance(broken_blocks, list) and len(broken_blocks) > 0 and isinstance(broken_blocks[0], list):
                # Multiple broken blocks; take the first one
                broken_blocks = broken_blocks[0]

            if broken_blocks[0] == 1 and broken_blocks[2] in [-63, -62]:  # Index 2 is y coordinate
                reward += self.block_break_reward

            # Additional reward for specific conditions
            if action_name == "attack":
                hand = self.normalize_hand(state)
                target_block = self.normalize_target_block(state)[0]  # Get scalar value
                if hand[0] == 1.0 and target_block == 1.0:
                    reward += 0.3

            tasks_norm = self.task  # Since task is constant
            blocks_norm = self.normalize_blocks(broken_blocks)
            hand_norm = self.normalize_hand(state)
            target_block_norm = self.normalize_target_block(state)
            flattened_matrix_norm = self.flatten_surrounding_blocks(state)

            state_data = {
                'image': screenshot,
                'tasks': tasks_norm,
                'blocks': blocks_norm,
                'hand': hand_norm,
                'target_block': target_block_norm,
                'flattened_matrix': flattened_matrix_norm
            }
        else:
            logging.warning("No state received after action.")
            state_data = self._get_default_state()

        self.steps += 1

        # Prepare combined_observation
        combined_observation = state_data

        # Split done into terminated and truncated
        terminated = self.steps >= self.max_episode_steps
        truncated = False
        info = {'reward_breakdown': {}}

        return combined_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Gym synchronous reset method"""
        if seed is not None:
            np.random.seed(seed)

        # Schedule the coroutine and get a Future
        future = asyncio.run_coroutine_threadsafe(
            self._async_reset(),
            self.loop
        )

        try:
            # Wait for the result with a timeout if necessary
            result = future.result(timeout=2)  # Adjust timeout as needed
            return result
        except Exception as e:
            logging.error(f"Error during reset: {e}")
            raise e

    async def _async_reset(self):
        """Asynchronous reset implementation"""
        self.steps = 0
        self.cumulative_rewards = 0.0
        self.episode_counts = 0

        try:
            # Clear the state queue
            while not self.state_queue.empty():
                self.state_queue.get_nowait()

            if not self.connected or self.websocket is None:
                logging.error("WebSocket not connected.")
                state_data = self._get_default_state()
                return state_data, {}

            # Send reset action
            await self.send_action("reset 2")

            # Receive state with timeout
            try:
                state = await asyncio.wait_for(self.state_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                logging.warning("Reset: No state received.")
                state_data = self._get_default_state()
                return state_data, {}

            # Capture screenshot asynchronously
            screenshot = await asyncio.get_event_loop().run_in_executor(None, self.capture_screenshot)

            # Process state
            if state is not None:
                tasks_norm = self.task
                broken_blocks = state.get('broken_blocks', [0, 0, 0, 0])
                blocks_norm = self.normalize_blocks(broken_blocks)
                hand_norm = self.normalize_hand(state)
                target_block_norm = self.normalize_target_block(state)
                flattened_matrix_norm = self.flatten_surrounding_blocks(state)

                state_data = {
                    'image': screenshot,
                    'tasks': tasks_norm,
                    'blocks': blocks_norm,
                    'hand': hand_norm,
                    'target_block': target_block_norm,
                    'flattened_matrix': flattened_matrix_norm
                }
            else:
                state_data = self._get_default_state()

            return state_data, {}

        except Exception as e:
            logging.error(f"Reset error: {e}")
            state_data = self._get_default_state()
            return state_data, {}

    def capture_screenshot(self):
        """Capture a screenshot of the assigned Minecraft window"""
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(self.minecraft_bounds)
                img = np.array(screenshot)[:, :, :3]  # Ensure RGB
                img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
                img = img.transpose(2, 0, 1) / 255.0  # Normalize

                if self.save_screenshots:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
                    cv2.imwrite(screenshot_path, img.transpose(1, 2, 0) * 255)
        except Exception as e:
            logging.error(f"Error capturing screenshot: {e}")
            img = np.zeros((3, 120, 120), dtype=np.float32)
        return img.astype(np.float32)

    def _get_default_state(self):
        """Return default state when real state cannot be obtained"""
        default = {
            'image': np.zeros((3, 120, 120), dtype=np.float32),
            'tasks': self.task,  # tasks_dim = 10
            'blocks': np.zeros(4, dtype=np.float32),  # block_features = 4
            'hand': np.zeros(5, dtype=np.float32),    # hand_dim = 5
            'target_block': np.zeros(1, dtype=np.float32),  # target_block_dim = 1
            'flattened_matrix': np.zeros(729, dtype=np.float32)  # flattened_matrix_dim = 729
        }
        return default

    def render(self, mode='human'):
        """Render the environment if needed."""
        pass

    def close(self):
        """Close the WebSocket connection and stop the thread"""
        if self.connected and self.websocket:
            self.connected = False
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            self.websocket = None

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=1)

    def __del__(self):
        """Cleanup before deleting the environment object."""
        self.close()

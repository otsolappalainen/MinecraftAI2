# File: data_collection_imitation_compatible.py

import asyncio
import websockets
import mss
import numpy as np
from PIL import Image
import time
import os
import threading
from pynput import keyboard, mouse
import json
import pickle
import logging
import pygetwindow as gw
import pygame
from queue import Queue, Empty
import fnmatch
import ctypes
import random  # For task selection

# ------------------------------
# Configuration and Setup
# ------------------------------

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set data directory for saving data
DATA_DIR = 'expert_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Create a unique session directory
SESSION_ID = time.strftime('%Y%m%d_%H%M%S')
SESSION_DIR = os.path.join(DATA_DIR, f'session_{SESSION_ID}')
os.makedirs(SESSION_DIR, exist_ok=True)

# Action mapping: index to action name
ACTION_MAPPING = {
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

# Reverse mapping: action name to index
ACTION_NAME_TO_INDEX = {v: k for k, v in ACTION_MAPPING.items()}

# Keyboard action keys mapping: key to action name
ACTION_KEYS = {
    'w': 'move_forward',
    's': 'move_backward',
    'a': 'move_left',
    'd': 'move_right',
    'space': 'jump',
    'shift': 'sneak',
    'q': 'jump_walk_forward',
    'e': 'next_item',
    'r': 'previous_item'
}

# Mouse movement threshold (pixels)
MOUSE_MOVE_THRESHOLD = 2  # Lower threshold for higher sensitivity

# Variables to store the latest inputs

latest_mouse_action = None
latest_mouse_action_time = None

# Thread-safe queue for mouse actions from the MouseCaptureWindow
mouse_action_queue = Queue()
pressed_keys = set()
pressed_mouse_buttons = set()  # Add this new set

# Lock for synchronizing access to shared variables
input_lock = threading.Lock()

# Flag to control the main loop
running = True

# Global variables for data and listeners
data = []
keyboard_listener = None
mouse_listener = None
mouse_capture = None

# Task configuration
TASK_SIZE = 20  # Size of the task array

possible_tasks = [
    [1, 0] + [0]*(TASK_SIZE - 2),   # Positive X direction
    [-1, 0] + [0]*(TASK_SIZE - 2),  # Negative X direction
    [0, 1] + [0]*(TASK_SIZE - 2),   # Positive Z direction
    [0, -1] + [0]*(TASK_SIZE - 2),  # Negative Z direction
]

# Reward scaling factors
REWARD_SCALE_POSITIVE = 1.0
immediate_reward = 0  # Assuming immediate reward is 0

# ------------------------------
# Helper Functions and Classes
# ------------------------------

def find_minecraft_window():
    """
    Finds the Minecraft window and returns its bounding box.
    If not found, raises an exception.
    Searches for window titles matching patterns:
    - Starts with "Minecraft"
    - Contains "1.21.3"
    - Contains "Singleplayer"
    """
    patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
    windows = gw.getAllTitles()
    for title in windows:
        for pattern in patterns:
            if fnmatch.fnmatch(title, pattern):
                matched_windows = gw.getWindowsWithTitle(title)
                if matched_windows:
                    minecraft_window = matched_windows[0]
                    if (minecraft_window.isMinimized):
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

class MouseCaptureWindow(threading.Thread):
    """
    A separate thread that creates a pygame window to capture relative mouse movements.
    It sends actions ('look_left', 'look_right', etc.) to a queue based on mouse movement.
    """
    def __init__(self, action_queue, threshold=2, window_bounds=None):
        super().__init__()
        self.action_queue = action_queue
        self.threshold = threshold
        self.running = True
        self.daemon = True  # Daemonize thread to exit with the main program
        self.window_bounds = window_bounds  # Dictionary with 'left', 'top', 'width', 'height'

    def run(self):
        pygame.init()
        # Create a window the same size and position as the Minecraft window
        if self.window_bounds:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.window_bounds['left']},{self.window_bounds['top']}"
            window_size = (self.window_bounds['width'], self.window_bounds['height'])
        else:
            window_size = (200, 200)  # Default size if bounds not provided

        screen = pygame.display.set_mode(window_size, pygame.NOFRAME)
        pygame.display.set_caption('Mouse Capture Window')

        # Set window transparency (Windows only)
        if os.name == 'nt':
            hwnd = pygame.display.get_wm_info()['window']
            # Set window to be layered (transparent)
            extended_style_settings = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, extended_style_settings | 0x80000)
            # Set Layered window attributes, set the opacity to 1 (almost transparent)
            ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 1, 0x00000002)

        # Hide the mouse cursor and capture the mouse
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        clock = pygame.time.Clock()

        logging.info("MouseCaptureWindow: Initialized and capturing mouse movements.")

        # Initialize previous mouse position
        pygame.mouse.get_rel()  # Reset relative movement

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            # Get relative mouse movement
            dx, dy = pygame.mouse.get_rel()
            if abs(dx) > self.threshold or abs(dy) > self.threshold:
                # Determine dominant movement axis
                if abs(dx) > abs(dy):
                    if dx > 0:
                        self.action_queue.put('look_right')
                        logging.debug(f"MouseCaptureWindow: Detected 'look_right' with dx={dx}")
                    else:
                        self.action_queue.put('look_left')
                        logging.debug(f"MouseCaptureWindow: Detected 'look_left' with dx={dx}")
                else:
                    if dy > 0:
                        self.action_queue.put('look_down')
                        logging.debug(f"MouseCaptureWindow: Detected 'look_down' with dy={dy}")
                    else:
                        self.action_queue.put('look_up')
                        logging.debug(f"MouseCaptureWindow: Detected 'look_up' with dy={dy}")
            clock.tick(60)  # Limit to 60 FPS

        pygame.quit()
        logging.info("MouseCaptureWindow: Shutting down.")

# ------------------------------
# Keyboard and Mouse Listeners
# ------------------------------

# Keyboard listener callbacks
def on_press(key):
    global running
    with input_lock:
        try:
            key_char = key.char.lower()
            if key_char == 'o':
                # Initiate graceful shutdown
                running = False
                logging.info("Graceful shutdown initiated by pressing 'o'.")
                return False  # Stop the listener
            if key_char in ACTION_KEYS:
                pressed_keys.add(key_char)
                logging.debug(f"Keyboard Listener: Key '{key_char}' pressed.")
        except AttributeError:
            # Handle special keys
            if key == keyboard.Key.space:
                pressed_keys.add('space')
                logging.debug("Keyboard Listener: 'space' key pressed.")
            elif key == keyboard.Key.shift:
                pressed_keys.add('shift')
                logging.debug("Keyboard Listener: 'shift' key pressed.")

def on_release(key):
    with input_lock:
        try:
            key_char = key.char.lower()
            if key_char in pressed_keys:
                pressed_keys.discard(key_char)
                logging.debug(f"Keyboard Listener: Key '{key_char}' released.")
        except AttributeError:
            if key == keyboard.Key.space and 'space' in pressed_keys:
                pressed_keys.discard('space')
                logging.debug("Keyboard Listener: 'space' key released.")
            elif key == keyboard.Key.shift and 'shift' in pressed_keys:
                pressed_keys.discard('shift')
                logging.debug("Keyboard Listener: 'shift' key released.")

# Mouse listener callbacks
def on_click(x, y, button, pressed):
    global pressed_mouse_buttons
    with input_lock:
        if button == mouse.Button.left:
            if pressed:
                pressed_mouse_buttons.add('attack')
                logging.debug("Mouse: Added 'attack' action")
            else:
                pressed_mouse_buttons.discard('attack')
                logging.debug("Mouse: Removed 'attack' action")
        elif button == mouse.Button.right:
            if pressed:
                pressed_mouse_buttons.add('use')
                logging.debug("Mouse: Added 'use' action")
            else:
                pressed_mouse_buttons.discard('use')
                logging.debug("Mouse: Removed 'use' action")

def on_scroll(x, y, dx, dy):
    pass  # Not used in this script

# ------------------------------
# get_action_index Function
# ------------------------------

def get_action_index(action_name):
    if action_name is None:
        logging.info("Action name is None, defaulting to 'no_op'.")
        return ACTION_NAME_TO_INDEX["no_op"]
    elif action_name in ACTION_NAME_TO_INDEX:
        return ACTION_NAME_TO_INDEX[action_name]
    else:
        logging.info(f"Unknown action '{action_name}', defaulting to 'no_op'.")
        return ACTION_NAME_TO_INDEX["no_op"]

# Add before main():
def get_current_action():
    with input_lock:
        # Debug current state
        logging.debug(f"Current pressed_mouse_buttons: {pressed_mouse_buttons}")
        
        # First check mouse queue since movement is high priority
        try:
            mouse_action = mouse_action_queue.get_nowait()
            logging.debug(f"Got mouse movement action: {mouse_action}")
            return mouse_action
        except Empty:
            # Then check mouse buttons
            if 'attack' in pressed_mouse_buttons:
                return 'attack'
            if 'use' in pressed_mouse_buttons:
                return 'use'
                
            # Finally check keyboard
            if 'w' in pressed_keys:
                if 'space' in pressed_keys:
                    return 'jump_walk_forward'
                return 'move_forward'
            if 's' in pressed_keys:
                return 'move_backward'
            if 'a' in pressed_keys:
                return 'move_left'
            if 'd' in pressed_keys:
                return 'move_right'
            if 'space' in pressed_keys:
                return 'jump'
            if 'shift' in pressed_keys:
                return 'sneak'
                
            return 'no_op'

# ------------------------------
# Main Data Collection Loop
# ------------------------------

async def main():
    global running, latest_mouse_action, latest_mouse_action_time
    global data, keyboard_listener, mouse_listener, mouse_capture

    # Start keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    keyboard_listener.start()
    logging.info("Keyboard Listener: Started.")

    # Start mouse listener
    mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    mouse_listener.start()
    logging.info("Mouse Listener: Started.")

    # Randomly select a task for the session
    task = random.choice(possible_tasks)
    task = np.array(task, dtype=np.float32)  # Convert to numpy array
    task_normalized = task[:2] / np.linalg.norm(task[:2]) if np.linalg.norm(task[:2]) != 0 else np.zeros(2)

    logging.info(f"Selected task: {task}")

    # Initialize data
    data = []
    try:
        # Connect to WebSocket server
        uri = "ws://localhost:8080"  # Update with the correct port if necessary
        async with websockets.connect(uri) as websocket:
            iteration = 0
            with mss.mss() as sct:
                # Attempt to find the Minecraft window
                try:
                    minecraft_bounds = find_minecraft_window()
                    logging.info(f"Minecraft window found: {minecraft_bounds}")
                except Exception as e:
                    logging.error(str(e))
                    # Fallback to full screen
                    monitor_info = sct.monitors[1]
                    logging.info("Falling back to full screen capture.")
                    minecraft_bounds = {
                        "left": monitor_info["left"],
                        "top": monitor_info["top"],
                        "width": monitor_info["width"],
                        "height": monitor_info["height"],
                    }

                # Start mouse capture window
                mouse_capture = MouseCaptureWindow(
                    action_queue=mouse_action_queue,
                    threshold=MOUSE_MOVE_THRESHOLD,
                    window_bounds=minecraft_bounds
                )
                mouse_capture.start()
                logging.info("MouseCaptureWindow: Started.")

                latest_mouse_action_time = None
                MOUSE_ACTION_DURATION = 0.1  # Keep mouse action for 0.1 seconds

                prev_x = None
                prev_z = None

                # Initialize previous observation for next_obs
                prev_observation = None

                while running:
                    try:
                        start_time = time.time()

                        # Process mouse movement - keep only latest action
                        latest_mouse_action = None
                        while not mouse_action_queue.empty():
                            latest_mouse_action = mouse_action_queue.get_nowait()
                            logging.debug(f"Main Loop: Got mouse action '{latest_mouse_action}'")

                        # Determine action with latest mouse movement
                        if latest_mouse_action:
                            action_name = latest_mouse_action
                        else:
                            action_name = get_current_action()
                        
                        logging.debug(f"Selected action: {action_name}")

                        # Send action to server
                        message = {'action': action_name}
                        await websocket.send(json.dumps(message))

                        # Check if there are mouse actions from the queue
                        try:
                            while not mouse_action_queue.empty():
                                mouse_action = mouse_action_queue.get_nowait()
                                with input_lock:
                                    latest_mouse_action = mouse_action
                                    latest_mouse_action_time = time.time()
                                    logging.debug(f"Main Loop: Received mouse action '{mouse_action}' from queue.")
                        except Exception as e:
                            logging.error(f"Error reading from mouse action queue: {e}")

                        # Determine the current action
                        action_name = get_current_action()
                        logging.debug(f"Current action selected: {action_name}")

                        # Get the action index
                        action_index = get_action_index(action_name)
                        action = ACTION_MAPPING[action_index]

                        # Send action to mod
                        message = {'action': action}
                        try:
                            await websocket.send(json.dumps(message))
                        except Exception as e:
                            logging.error(f"Error sending action to mod: {e}")

                        # Capture screenshot
                        try:
                            sct_img = sct.grab(minecraft_bounds)
                            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
                        except Exception as e:
                            logging.error(f"Error capturing Minecraft window: {e}")
                            # Create a black image as fallback
                            img = Image.new('RGB', (224, 224), color='black')

                        img = img.resize((224, 224))
                        image_array = np.array(img).transpose((2, 0, 1))  # Shape: (3, 224, 224)

                        # Receive state from mod
                        try:
                            response = await websocket.recv()
                            state = json.loads(response)
                            logging.debug(f"Received state: {state}")
                        except Exception as e:
                            logging.error(f"Error receiving state from mod: {e}")
                            state = {}

                        # Extract position
                        x = state.get('x', 0)
                        y_coord = state.get('y', 0)
                        z = state.get('z', 0)

                        # Calculate movement vector
                        if prev_x is not None and prev_z is not None:
                            delta_x = x - prev_x
                            delta_z = z - prev_z
                            movement_vector = np.array([delta_x, delta_z], dtype=np.float32)
                            # Calculate movement alignment
                            movement_alignment = np.dot(movement_vector, task_normalized)
                            # Calculate reward
                            reward = immediate_reward + (movement_alignment * REWARD_SCALE_POSITIVE)
                            logging.info(f"Iteration {iteration}: Reward calculated: {reward}")
                        else:
                            movement_vector = np.array([0.0, 0.0], dtype=np.float32)
                            reward = immediate_reward
                            logging.info(f"Iteration {iteration}: Initial step, reward set to {reward}")

                        # Update previous position
                        prev_x = x
                        prev_z = z

                        # Normalize state variables
                        yaw = state.get('yaw', 0)
                        health = state.get('health', 20)
                        hunger = state.get('hunger', 20)
                        alive = float(state.get('alive', True))

                        # Normalize yaw to [-180, 180] first
                        yaw = ((yaw + 180) % 360) - 180

                        # Then normalize yaw to [-1, 1]
                        normalized_yaw = yaw / 180.0

                        # Other normalizations
                        normalized_x = x / 20000.0
                        normalized_y = y_coord / 20000.0 
                        normalized_z = z / 256.0
                        sin_yaw = np.sin(np.deg2rad(yaw))  # Keep these for additional features
                        cos_yaw = np.cos(np.deg2rad(yaw))
                        normalized_health = health / 20.0
                        normalized_hunger = hunger / 20.0

                        # Process broken blocks
                        max_broken_blocks = 5  # Maximum number of broken blocks to track
                        broken_blocks_dim = 4  # [blocktype, x, y, z] per block
                        broken_blocks_array = np.zeros(max_broken_blocks * broken_blocks_dim, dtype=np.float32)

                        # Fill array with normalized block data
                        broken_blocks = state.get('broken_blocks', [])
                        for i, block in enumerate(broken_blocks[:max_broken_blocks]):
                            idx = i * broken_blocks_dim
                            # Normalize block type (currently always 1)
                            broken_blocks_array[idx] = block.get('blocktype', 0) / 10.0  
                            # Normalize coordinates similar to player position
                            broken_blocks_array[idx + 1] = block.get('blockx', 0) / 20000.0
                            broken_blocks_array[idx + 2] = block.get('blocky', 0) / 20000.0
                            broken_blocks_array[idx + 3] = block.get('blockz', 0) / 256.0

                        # Add broken blocks to other observations
                        other = np.concatenate([
                            np.array([
                                normalized_x,
                                normalized_z, 
                                normalized_y,
                                sin_yaw,
                                cos_yaw,
                                normalized_health,
                                normalized_hunger,
                                alive
                            ], dtype=np.float32),
                            broken_blocks_array
                        ])

                        observation = {
                            'image': image_array.astype(np.float32) / 255.0,
                            'other': other,
                            'task': task
                        }

                        # For BC, we need to store observations, actions, next_observations, dones, infos
                        # Prepare next_observation for the previous step
                        if prev_observation is not None:
                            data_entry = {
                                'observation': prev_observation,
                                'action': prev_action_index,
                                'next_observation': observation,
                                'done': False,  # Assuming continuous task
                                'info': {}  # Additional info if needed
                            }
                            data.append(data_entry)

                        # Update previous observation and action
                        prev_observation = observation
                        prev_action_index = action_index

                        iteration += 1

                        # Wait for next iteration
                        # Run at 20Hz
                        elapsed_time = time.time() - start_time
                        sleep_time = max(0, 0.05 - elapsed_time)

                        await asyncio.sleep(sleep_time)
                    except websockets.exceptions.ConnectionClosed:
                        logging.info("WebSocket connection closed")
                        break
                    except Exception as e:
                        logging.error(f"Error in main loop: {e}")
                        break

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Graceful shutdown sequence
        running = False
        logging.info("Starting graceful shutdown...")

        # Stop input listeners
        if keyboard_listener:
            keyboard_listener.stop()
        if mouse_listener:
            mouse_listener.stop()

        # Stop mouse capture window
        if mouse_capture and mouse_capture.is_alive():
            logging.info("Stopping MouseCaptureWindow...")
            mouse_capture.running = False
            mouse_capture.join(timeout=2)
            
        # Save collected data
        if data:
            try:
                with open(os.path.join(SESSION_DIR, 'expert_data.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                logging.info(f"Saved data to {SESSION_DIR}/expert_data.pkl")
            except Exception as e:
                logging.error(f"Error saving data: {e}")

        logging.info("Shutdown complete")

# Add signal handler
def signal_handler(sig, frame):
    global running
    logging.info("Received shutdown signal")
    running = False

if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(main())

# Global variables
mouse_action_queue = Queue()  # Re-add queue for mouse movement
pressed_mouse_buttons = set() 
pressed_keys = set()
input_lock = threading.Lock()

class MouseCaptureWindow(threading.Thread):
    def __init__(self, action_queue, threshold=2, window_bounds=None):
        super().__init__()
        self.action_queue = action_queue  # Queue for look actions
        self.threshold = threshold
        self.running = True
        self.daemon = True
        self.window_bounds = window_bounds

    def run(self):
        # ... existing MouseCaptureWindow init code ...
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            # Get relative mouse movement
            dx, dy = pygame.mouse.get_rel()
            if abs(dx) > self.threshold or abs(dy) > self.threshold:
                if abs(dx) > abs(dy):
                    action = 'look_right' if dx > 0 else 'look_left'
                else:
                    action = 'look_down' if dy > 0 else 'look_up'
                self.action_queue.put(action)
                logging.debug(f"MouseCaptureWindow: Added '{action}' to queue")

def get_current_action():
    with input_lock:
        logging.debug(f"Current pressed_mouse_buttons: {pressed_mouse_buttons}")
        
        # Priority 1: Mouse buttons
        if 'attack' in pressed_mouse_buttons:
            return 'attack'
        if 'use' in pressed_mouse_buttons:
            return 'use'
            
        # Priority 2: Movement keys
        if 'w' in pressed_keys:
            if 'space' in pressed_keys:
                return 'jump_walk_forward'
            return 'move_forward'
        if 's' in pressed_keys:
            return 'move_backward'
        if 'a' in pressed_keys:
            return 'move_left'
        if 'd' in pressed_keys:
            return 'move_right'
        if 'space' in pressed_keys:
            return 'jump'
        if 'shift' in pressed_keys:
            return 'sneak'
            
        # Priority 3: Mouse movement
        try:
            return mouse_action_queue.get_nowait()
        except Empty:
            return 'no_op'

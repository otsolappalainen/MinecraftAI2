import pyautogui
import math
import time
import cv2
import numpy as np
from PIL import Image
import re
import torch
from torchvision import transforms
import torch.nn as nn
import win32api
import win32con
import ctypes
import mss  # For efficient screen capturing
import os

# Constants for mouse control
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
mouse_event = ctypes.windll.user32.mouse_event

# Define screen capture regions
COORD_REGION = {"x_offset": 66, "y_offset": 308, "width": 530, "height": 29}
ANGLE_REGION = {"x_offset": 519, "y_offset": 385, "width": 276, "height": 31}

# Define character classes and load the model
character_classes = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '(', ')', '/', ' ']
character_model_path = "character_recognition_model_8.pth"

# Character recognition model definition
class CharacterRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(CharacterRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the character recognition model
character_model = CharacterRecognitionModel(num_classes=len(character_classes))
character_model.load_state_dict(torch.load(character_model_path))
device = torch.device("cpu")  # Set device to CPU for small models
character_model = character_model.to(device)
character_model.eval()

# Define preprocessing for the character model
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def apply_binary_threshold(image_np, threshold=219):
    """Apply binary threshold and invert colors for character recognition."""
    # Apply binary thresholding to isolate characters
    _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    # Invert the image so characters are black on a white background
    inverted_image = cv2.bitwise_not(binary_image)
    return inverted_image

class MinecraftAgent:
    def __init__(self, screenshot_dir="screenshots"):
        self.timers = {}
        self.state = (0, 0, 0, 0, 0, None, 10.0, 10.0, True)  # Format: (x, y, z, yaw, pitch, processed_image, health, hunger, alive)
        self.previous_state = (0, 0, 0, 0, 0, None, 10.0, 10.0, True)
        self._camera_movement_active = False
        self.screenshot_dir = screenshot_dir
        os.makedirs(screenshot_dir, exist_ok=True) 

        # Movement flags
        self.moving_forward = False
        self.moving_backward = False
        self.strafing_left = False
        self.strafing_right = False
        self.is_sneaking = False
        self.is_jumping = False
        self.is_walking = False 
        self.is_left_clicking = False
        self.is_space_toggled = False


        # Initialize mss for screen capture
        self.sct = mss.mss()  # Keep mss instance open

    def capture_full_screen_raw(self):
        """Capture the full screen once and reuse the image."""
        monitor = self.sct.monitors[1]  # Full primary monitor
        screenshot = np.array(self.sct.grab(monitor))
        return screenshot

    def extract_region(self, full_screen_image, region):
        """Extract specific regions from the full screen image."""
        x_offset = region["x_offset"]
        y_offset = region["y_offset"]
        width = region["width"]
        height = region["height"]
        # Crop the region from the full screen image
        region_image = full_screen_image[y_offset:y_offset+height, x_offset:x_offset+width]
        grayscale_image = cv2.cvtColor(region_image, cv2.COLOR_BGRA2GRAY)
        return grayscale_image  # Return as NumPy array

    def process_full_screen(self, full_screen_image):
        """Process the full screen image into a tensor."""
        # Convert to grayscale
        grayscale_image = cv2.cvtColor(full_screen_image, cv2.COLOR_BGRA2GRAY)

        # Resize to 224x224 using CPU-based resize
        screenshot_resized = cv2.resize(grayscale_image, (224, 224), interpolation=cv2.INTER_AREA)

        # Normalize and convert to tensor directly
        screenshot_normalized = ((torch.from_numpy(screenshot_resized).float() / 255.0 - 0.5) * 2).to(device)
        tensor_image = screenshot_normalized.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        return tensor_image

    def process_image(self, image_np):
        """Process the image to extract text using batched predictions."""
        start_time = time.time()
        binary_image = apply_binary_threshold(image_np, threshold=219)
        height, width = binary_image.shape
        recognized_text = ""
        start_col = None
        last_end = None
        char_index = 0

        char_images = []
        space_indices = []

        for col in range(width):
            black_count = np.sum(binary_image[:, col] == 0)
            if black_count == 0:
                if start_col is not None:
                    char_image = binary_image[:, start_col:col]
                    char_images.append(char_image)
                    char_index += 1
                    start_col = None
                    last_end = col
            else:
                if start_col is None:
                    start_col = col

            if last_end is not None and start_col is not None and start_col - last_end >= 10:
                space_indices.append(len(char_images))
                last_end = None

        if start_col is not None:
            char_image = binary_image[:, start_col:width]
            char_images.append(char_image)

        # Batch predict characters
        recognized_chars = self.predict_characters_batch(char_images)

        # Insert spaces at appropriate positions
        for idx in space_indices:
            recognized_chars.insert(idx, ' ')

        recognized_text = ''.join(recognized_chars)
        self.timers['process_image'] = time.time() - start_time
        return recognized_text

    def predict_characters_batch(self, char_images):
        """Predict multiple characters in a batch."""
        image_tensors = []
        for char_image in char_images:
            char_pil = Image.fromarray(char_image)
            image_tensor = transform(char_pil)
            image_tensors.append(image_tensor)
        if len(image_tensors) == 0:
            return []
        image_tensors = torch.stack(image_tensors).to(device)
        with torch.no_grad():
            outputs = character_model(image_tensors)
            _, predicted_indices = torch.max(outputs, 1)
        recognized_chars = [character_classes[idx.item()] for idx in predicted_indices]
        #print(f"chars: {recognized_chars}")
        return recognized_chars

    def parse_coordinates(self, text):
        """Parse coordinates from recognized text."""
        try:
            # Directly format and clean text to match 'float / float / float'
            cleaned_text = ''.join(text).replace(" ", "")  # Joins list into string and removes spaces
            matches = re.findall(r"-?\d+\.\d+", cleaned_text)
            if len(matches) == 3:
                x = float(matches[0])
                y = float(matches[1])
                z = float(matches[2])
                return x, y, z
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
        return None, None, None

    def parse_angles(self, text):
        """Parse yaw and pitch from recognized text."""
        try:
            # Directly format and clean text to match '(float / float)'
            cleaned_text = ''.join(text).replace(" ", "").replace("(", "").replace(")", "")  # Format string
            match = re.match(r"(-?\d+\.\d+)\s*/\s*(-?\d+\.\d+)", cleaned_text)
            if match:
                yaw = float(match.group(1))
                pitch = float(match.group(2))
                return yaw, pitch
        except Exception as e:
            print(f"Error parsing angles: {e}")
        return 0, 0

    def get_state(self):
        """Get the current state by capturing the screen once and processing."""
        start_time = time.time()

        # Timing screen capture
        capture_start = time.time()
        full_screen_image = self.capture_full_screen_raw()
        self.timers['screen_capture'] = time.time() - capture_start

        # Timing region extraction
        extraction_start = time.time()
        coord_screenshot = self.extract_region(full_screen_image, COORD_REGION)
        angle_screenshot = self.extract_region(full_screen_image, ANGLE_REGION)
        self.timers['region_extraction'] = time.time() - extraction_start

        # Timing image processing
        processing_start = time.time()
        recognized_coords = self.process_image(coord_screenshot)
        recognized_angles = self.process_image(angle_screenshot)
        self.timers['image_processing'] = time.time() - processing_start

        # Timing parsing
        parsing_start = time.time()
        x, y, z = self.parse_coordinates(recognized_coords)
        yaw, pitch = self.parse_angles(recognized_angles)
        self.timers['parsing'] = time.time() - parsing_start

        # Timing full screen processing
        full_screen_processing_start = time.time()
        full_screen_screenshot = self.process_full_screen(full_screen_image)
        self.timers['full_screen_processing'] = time.time() - full_screen_processing_start

        # Health and hunger remain hardcoded for now
        health = 10.0
        hunger = 10.0
        alive = True  # Hardcoded alive status

        # Initialize variables with previous state or zero
        if self.previous_state:
            prev_x, prev_y, prev_z, prev_yaw, prev_pitch, prev_screenshot, prev_health, prev_hunger, prev_alive = self.previous_state
        else:
            prev_x, prev_y, prev_z, prev_yaw, prev_pitch = 0.0, 0.0, 0.0, 0.0, 0.0
            prev_screenshot = None
            prev_health, prev_hunger, prev_alive = 10.0, 10.0, True

        # Replace None values with previous or zero
        if x is None:
            print("Failed to parse x coordinate.")
            x = prev_x
        if y is None:
            print("Failed to parse y coordinate.")
            y = prev_y
        if z is None:
            print("Failed to parse z coordinate.")
            z = prev_z
        if yaw is None:
            print("Failed to parse yaw.")
            yaw = prev_yaw
        if pitch is None:
            print("Failed to parse pitch.")
            pitch = prev_pitch
        if full_screen_screenshot is None:
            print("Failed to process full screen screenshot.")
            full_screen_screenshot = prev_screenshot

        # Update state
        self.state = (x, y, z, yaw, pitch, full_screen_screenshot, health, hunger, alive)

        # Save current state as previous state for next iteration
        self.previous_state = self.state
        self.timers['get_state'] = time.time() - start_time

    def smooth_mouse_move(self, x_offset, y_offset, duration=0.1, steps=15):
        """Smoothly move the mouse by x_offset and y_offset over a given duration."""
        delay = duration / steps
        step_x = x_offset / steps
        step_y = y_offset / steps
        for _ in range(steps):
            mouse_event(MOUSEEVENTF_MOVE, int(step_x), int(step_y), 0, 0)
            time.sleep(delay)

    def stop_camera_movement(self):
        """Stops continuous camera movement by setting the active flag to False."""
        print("Stopping camera movement...")
        self._camera_movement_active = False
    
    def left_click(self, hold_duration=0.2):
        if not self.is_left_clicking:
            # Start left click
            print("Starting left click...")
            pyautogui.mouseDown(button='left')
            self.is_left_clicking = True
        else:
            # Stop left click
            print("Stopping left click...")
            pyautogui.mouseUp(button='left')
            self.is_left_clicking = False

    def right_click(self, hold_duration=0.1):
        print("Performing right click...")
        pyautogui.mouseDown(button='right')
        time.sleep(hold_duration)
        pyautogui.mouseUp(button='right')

    def get_position(self):
        x, y, z, yaw, pitch = self.state[:5]
        return (x, y, z, yaw, pitch)
    
    def set_position(self):
        print("bad set position called")
        x, y, z, yaw, pitch = self.state[:5]
        return (x, y, z, yaw, pitch)
    

    def toggle_space(self):
        """Toggle spacebar on or off."""
        if not self.is_space_toggled:
            pyautogui.keyDown('space')
            self.is_space_toggled = True
        else:
            pyautogui.keyUp('space')
            self.is_space_toggled = False

    def press_1(self):
        pyautogui.keyDown('1')
        time.sleep(0.05)  # Timer set to 0.05 seconds
        pyautogui.keyUp('1')

    def press_2(self):
        pyautogui.keyDown('2')
        time.sleep(0.05)
        pyautogui.keyUp('2')

    def press_3(self):
        pyautogui.keyDown('3')
        time.sleep(0.05)
        pyautogui.keyUp('3')

    def press_4(self):
        pyautogui.keyDown('4')
        time.sleep(0.05)
        pyautogui.keyUp('4')

    def press_5(self):
        pyautogui.keyDown('5')
        time.sleep(0.05)
        pyautogui.keyUp('5')

    def press_6(self):
        pyautogui.keyDown('6')
        time.sleep(0.05)
        pyautogui.keyUp('6')

    def press_7(self):
        pyautogui.keyDown('7')
        time.sleep(0.05)
        pyautogui.keyUp('7')

    def press_8(self):
        pyautogui.keyDown('8')
        time.sleep(0.05)
        pyautogui.keyUp('8')

    def press_9(self):
        pyautogui.keyDown('9')
        time.sleep(0.05)
        pyautogui.keyUp('9')

    

    # Movement functions
    def move_forward(self):
        if not self.moving_forward:
            pyautogui.keyDown('w')
            self.moving_forward = True

    def stop_moving_forward(self):
        if self.moving_forward:
            pyautogui.keyUp('w')
            self.moving_forward = False

    def move_backward(self):
        if not self.moving_backward:
            pyautogui.keyDown('s')
            self.moving_backward = True

    def stop_moving_backward(self):
        if self.moving_backward:
            pyautogui.keyUp('s')
            self.moving_backward = False

    def strafe_left(self):
        if not self.strafing_left:
            pyautogui.keyDown('a')
            self.strafing_left = True

    def stop_strafing_left(self):
        if self.strafing_left:
            pyautogui.keyUp('a')
            self.strafing_left = False

    def strafe_right(self):
        if not self.strafing_right:
            pyautogui.keyDown('d')
            self.strafing_right = True

    def stop_strafing_right(self):
        if self.strafing_right:
            pyautogui.keyUp('d')
            self.strafing_right = False

    def toggle_sneak(self):
        """Toggle sneak (crouch) on or off."""
        if not self.is_sneaking:
            pyautogui.keyDown('ctrl')
            self.is_sneaking = True
        else:
            pyautogui.keyUp('ctrl')
            self.is_sneaking = False

    # Jumping
    def jump(self):
        if not self.is_jumping:
            pyautogui.keyDown('space')
            self.is_jumping = True
            time.sleep(0.1)  # Adjust the sleep duration to control jump timing
            pyautogui.keyUp('space')
            self.is_jumping = False

    def toggle_walk(self):
        """Toggle walking forward on or off."""
        if not self.is_walking:
            pyautogui.keyDown('w')
            self.is_walking = True
            print("walking")
        else:
            pyautogui.keyUp('w')
            self.is_walking = False


    
    # Camera movement functions
    def smooth_mouse_move(self, x_offset, y_offset, duration=0.1, steps=15):
        delay = duration / steps
        step_x = x_offset / steps
        step_y = y_offset / steps
        for _ in range(steps):
            mouse_event(MOUSEEVENTF_MOVE, int(step_x), int(step_y), 0, 0)
            time.sleep(delay)

    def look_up(self, speed=200):
        self.smooth_mouse_move(0, -speed)

    def look_down(self, speed=200):
        self.smooth_mouse_move(0, speed)

    def turn_left(self, speed=200, big=False):
        offset = -speed * (5 if big else 1)
        self.smooth_mouse_move(offset, 0)

    def turn_right(self, speed=200, big=False):
        offset = speed * (5 if big else 1)
        self.smooth_mouse_move(offset, 0)

    def big_turn_left(self, speed=200):
        offset = -speed * 5
        self.smooth_mouse_move(offset, 0)

    def big_turn_right(self, speed=200):
        offset = speed * 5
        self.smooth_mouse_move(offset, 0)



    

    def stop_camera_movement(self):
        print("Stopping camera movement...")
        self._camera_movement_active = False

    # Stop all movements
    def stop_all_movements(self):
        # Stop all movement actions
        self.stop_moving_forward()
        self.stop_moving_backward()
        self.stop_strafing_left()
        self.stop_strafing_right()
        if self.is_sneaking:
            self.toggle_sneak()
        if self._camera_movement_active:
            self.stop_camera_movement()
        if self.is_walking:
            self.toggle_walk()
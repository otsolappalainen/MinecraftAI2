import pyautogui
import math
import time
import cv2
import numpy as np
from PIL import Image, ImageGrab
import re
import torch
from torchvision import transforms
import torch.nn as nn
import win32api
import win32con
import ctypes

# Constants for the mouse_event function
MOUSEEVENTF_MOVE = 0x0001  # Move the mouse
MOUSEEVENTF_ABSOLUTE = 0x8000  # Move is based on absolute position

# Access to mouse_event from the user32 library
mouse_event = ctypes.windll.user32.mouse_event


# Define the character classes
character_classes = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '(', ')', '/', ' ']

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
character_model_path = "character_recognition_model_8.pth"
character_model.load_state_dict(torch.load(character_model_path))
character_model.eval()

# Preprocessing function for the character model
def apply_binary_threshold(image, threshold=200):
    """Apply binary threshold to enhance contrast for character recognition."""
    image_np = np.array(image)
    _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_image)

transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_binary_threshold(img, threshold=200)),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_character(image):
    """Predict a single character from an image."""
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = character_model(image_tensor)
        _, predicted = torch.max(output, 1)
    return character_classes[predicted.item()]

def capture_region(region):
    """Capture a specific region of the screen."""
    bbox = (region["x_offset"], region["y_offset"], 
            region["x_offset"] + region["width"], 
            region["y_offset"] + region["height"])
    screenshot = ImageGrab.grab(bbox=bbox).convert("L")  # Convert to grayscale
    return screenshot

def process_image(image):
    """Split the captured image into characters and predict each."""
    binary_image = np.array(image)
    _, binary_image = cv2.threshold(binary_image, 200, 255, cv2.THRESH_BINARY_INV)
    height, width = binary_image.shape

    recognized_text = ""
    start_col = None
    last_end = None

    for col in range(width):
        black_count = np.sum(binary_image[:, col] == 0)  # Count black pixels

        if black_count == 0:
            if start_col is not None:
                char_image = binary_image[:, start_col:col]
                char_pil = Image.fromarray(char_image)
                recognized_text += predict_character(char_pil)
                start_col = None
                last_end = col
        else:
            if start_col is None:
                start_col = col

        if last_end is not None and start_col is not None and start_col - last_end >= 10:
            recognized_text += " "
            last_end = None

    if start_col is not None:
        char_image = binary_image[:, start_col:width]
        char_pil = Image.fromarray(char_image)
        recognized_text += predict_character(char_pil)

    #print(f"recognized test: {recognized_text}")

    return recognized_text

def parse_coordinates(text):
    """Parse coordinates from recognized text."""
    try:
        cleaned_text = text.replace("slash", "/").replace("dot", ".").replace("dash", "-").strip()
        matches = re.findall(r"-?\d+\.\d+|-?\d+", cleaned_text)
        if len(matches) == 3:
            x = float(matches[0])
            y = float(matches[1])
            z = float(matches[2])
            print(f"x: {x}, y: {y}, z: {z}")
            return x, y, z
    except Exception:
        pass  # Silently ignore errors
    return None, None, None

def parse_angles(text):
    """Parse yaw (horizontal angle) and pitch (vertical angle) from text."""
    try:
        # Clean up any leading or trailing characters not part of the coordinates
        cleaned_text = text.strip().replace(") (", "(").replace(")", "").replace("(", "")
        
        # Regular expression to capture two floats separated by a slash
        match = re.match(r"(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)", cleaned_text)
        if match:
            # Extract yaw and pitch
            yaw = float(match.group(1))
            pitch = float(match.group(2))
            print(f"Parsed angles - Yaw: {yaw}, Pitch: {pitch}")
            return yaw, pitch
    except Exception as e:
        print(f"Error parsing angles: {e}")
    
    # Return default values if parsing fails
    return 0, 0


class MinecraftAgent:
    def __init__(self):
        self.actions = ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop']
        self.state = (0, 0, 0, 0)
        self.previous_state = (0, 0, 0, 0)

    
    def smooth_mouse_move(self, x_offset, y_offset, duration=0.1, steps=15):
        """
        Smoothly move the mouse by x_offset and y_offset over a given duration using `mouse_event`.

        Parameters:
        - x_offset: Total horizontal movement in pixels.
        - y_offset: Total vertical movement in pixels.
        - duration: Total time for the movement in seconds.
        - steps: Number of steps to divide the movement into.
        """
        # Calculate the delay and per-step movement
        delay = duration / steps
        step_x = x_offset / steps
        step_y = y_offset / steps

        # Perform the smooth movement by sending incremental signals
        for i in range(steps):
            # Use mouse_event to move the mouse by a small amount
            mouse_event(MOUSEEVENTF_MOVE, int(step_x), int(step_y), 0, 0)

            # Pause briefly to create the smooth effect
            time.sleep(delay)



    def get_state(self):
        # Define regions for position coordinates and angles
        region = {"x_offset": 66, "y_offset": 308, "width": 530, "height": 29}
        angle = {"x_offset": 519, "y_offset": 385, "width": 276, "height": 31}
        
        # Capture the screen regions
        screenshot = capture_region(region)
        angle_screenshot = capture_region(angle)
        
        # Process the images to extract text
        recognized_text = process_image(screenshot)
        recognized_angle = process_image(angle_screenshot)
        
        # Parse coordinates and angles
        x, y, z = parse_coordinates(recognized_text)
        yaw, pitch = parse_angles(recognized_angle)
        
        print(f"Parsed position: X={x}, Y={y}, Z={z} | Parsed angles: Yaw={yaw}, Pitch={pitch}")

        # Store both position and angle in the agent's state
        if x is not None and z is not None:
            # Update state and previous_state if parsing succeeded
            self.state = (x, z, yaw, pitch)
            self.previous_state = self.state
        else:
            # Fallback to the last known state
            print("Failed to read coordinates. Falling back to the previous state.")
            self.state = self.previous_state
        
        print(f"Final state set in get_state: {self.state}")


    def perform_action(self, action):
        print("Performing action:", action)
        if action == 'move_forward':
            self.move_forward()
        elif action == 'move_backward':
            self.move_backward()
        elif action == 'strafe_left':
            self.strafe_left()
        elif action == 'strafe_right':
            self.strafe_right()
        elif action == 'turn_left':
            self.turn_left()
        elif action == 'turn_right':
            self.turn_right()
        print(f"Agent state after action {action}: {self.state}")

    def move_forward(self):
        print("Moving forward...")
        # Implement forward movement logic here, e.g., using pyautogui:
        pyautogui.keyDown('w')
        time.sleep(0.1)
        pyautogui.keyUp('w')

    def move_backward(self):
        print("Moving backward...")
        pyautogui.keyDown('s')
        time.sleep(0.1)
        pyautogui.keyUp('s')

    def strafe_left(self):
        print("Strafing left...")
        pyautogui.keyDown('a')
        time.sleep(0.1)
        pyautogui.keyUp('a')

    def strafe_right(self):
        print("Strafing right...")
        pyautogui.keyDown('d')
        time.sleep(0.1)
        pyautogui.keyUp('d')

    def turn_left(self):
        print("Turning left smoothly...")
        self.smooth_mouse_move(-200, 0)  # Move left by 200 pixels

    def turn_right(self):
        print("Turning right smoothly...")
        self.smooth_mouse_move(200, 0)  # Move right by 200 pixels
    
    


import cv2
import numpy as np
import torch
import time
from PIL import Image, ImageGrab
from torchvision import transforms
import torch.nn as nn
import os
import re

# Path to the character recognition model
character_model_path = "character_recognition_model_6.pth"

# Define character classes
character_classes = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/', ' ']

# Load the character recognition model
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

# Instantiate and load the character recognition model
character_model = CharacterRecognitionModel(num_classes=len(character_classes))
character_model.load_state_dict(torch.load(character_model_path))
character_model.eval()

# Preprocessing for character model
def apply_binary_threshold(image, threshold=200):
    """Apply binary threshold to enhance contrast for character recognition."""
    image_np = np.array(image)
    _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_image)

transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_binary_threshold(img, threshold=200)),  # Apply binary threshold
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to predict a single character
def predict_character(image):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = character_model(image_tensor)
        _, predicted = torch.max(output, 1)
    return character_classes[predicted.item()]

# Capture region of the screen
def capture_region(region):
    bbox = (region["x_offset"], region["y_offset"], 
            region["x_offset"] + region["width"], 
            region["y_offset"] + region["height"])
    screenshot = ImageGrab.grab(bbox=bbox).convert("L")  # Convert to grayscale
    return screenshot

# Split captured image into characters and predict
def process_image(image):
    """Split the captured image into characters, predict each, and reconstruct coordinates."""
    # Convert to binary for easier character detection
    binary_image = np.array(image)
    _, binary_image = cv2.threshold(binary_image, 200, 255, cv2.THRESH_BINARY_INV)
    height, width = binary_image.shape

    # Detect character segments
    recognized_text = ""
    start_col = None
    last_end = None

    for col in range(width):
        black_count = np.sum(binary_image[:, col] == 0)  # Count black pixels

        if black_count == 0:
            if start_col is not None:
                # Segment and predict character
                char_image = binary_image[:, start_col:col]
                char_pil = Image.fromarray(char_image)
                recognized_text += predict_character(char_pil)
                start_col = None
                last_end = col
        else:
            if start_col is None:
                start_col = col

        # Add space for gaps of at least 10 pixels
        if last_end is not None and start_col is not None and start_col - last_end >= 10:
            recognized_text += " "
            last_end = None  # Reset after adding space

    # Predict the last character if the image ends with one
    if start_col is not None:
        char_image = binary_image[:, start_col:width]
        char_pil = Image.fromarray(char_image)
        recognized_text += predict_character(char_pil)

    return recognized_text

# Convert the predicted text into (x, y, z) coordinates


def parse_coordinates(text):
    print(f"Parsing text: '{text}'")  # Debug parsing input

    # Clean up text by replacing recognized words with symbols and removing extra spaces
    cleaned_text = text.replace("slash", "/").replace("dot", ".").replace("dash", "-").strip()

    # Use regex to match three parts that look like numbers, possibly separated by slashes or spaces
    matches = re.findall(r"-?\d+\.\d+|-?\d+", cleaned_text)

    # Check if we have exactly three coordinate parts
    if len(matches) == 3:
        try:
            x = float(matches[0])
            y = float(matches[1])
            z = float(matches[2])
            return x, y, z
        except ValueError as e:
            print(f"ValueError during parsing: {e}")

    print("Failed to parse coordinates.")
    return None, None, None


def main():
    region = {"x_offset": 66, "y_offset": 308, "width": 530, "height": 29}
    print("Starting screen capture every 5 seconds using the character model. Press Ctrl+C to stop.")
    try:
        while True:
            screenshot = capture_region(region)
            recognized_text = process_image(screenshot)
            print(f"Recognized Text: {recognized_text}")
            x, y, z = parse_coordinates(recognized_text)
            print(f"Predicted Coordinates (x, y, z): ({x}, {y}, {z})")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Screen capture stopped.")

if __name__ == "__main__":
    main()

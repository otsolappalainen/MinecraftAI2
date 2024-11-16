import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Directory with the input images
input_dir = r"C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\minecraft_screenshots"

# Define character classes
character_classes = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/', ' ']

# Load the trained model
class CharacterRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CharacterRecognitionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model_path = "character_recognition_model_6.pth"
model = CharacterRecognitionModel(num_classes=len(character_classes))
model.load_state_dict(torch.load(model_path))
model.eval()

# Custom binary threshold transformation
def apply_binary_threshold(image, threshold=200):
    """Apply binary threshold to enhance contrast for character recognition."""
    image_np = np.array(image)
    _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_image)

# Transformation pipeline for the model
transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_binary_threshold(img, threshold=200)),  # Apply binary threshold
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
])

def predict_character(image):
    """Predict the character in the given image segment using the trained model."""
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return character_classes[predicted.item()]

def process_image(image_path):
    """Process a single image to detect characters and predict the full text."""
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
        return ""
    
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    height, width = binary_image.shape

    # Detect character segments
    recognized_text = ""
    start_col = None
    last_end = None

    for col in range(width):
        black_count = np.sum(binary_image[:, col] == 0)  # Count black pixels in the column

        if black_count == 0:
            if start_col is not None:
                # Segment a character region and predict it
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

# Main function to process all images in the input directory
def main():
    for filename in sorted(os.listdir(input_dir)):
        image_path = os.path.join(input_dir, filename)
        recognized_text = process_image(image_path)
        print(f"Image: {filename} -> Recognized Text: {recognized_text}")

if __name__ == "__main__":
    main()

import cv2
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Directory paths
image_dir = "minecraft_screenshots"  # Folder with the original screenshots

# Adjustable parameters
window_width = 20  # Width of the sliding window
window_height = 30  # Height of the sliding window
step_size = 1       # Step size for sliding the window
binary_threshold = 200  # Threshold for binary conversion

# Character classes (adjust this list based on your dataset folder names)
character_classes = ['dash','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'slash', 'space']

# Model definition - simple CNN
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

# Load the trained model
num_classes = len(character_classes)
model = CharacterRecognitionModel(num_classes)
model.load_state_dict(torch.load("character_recognition_model_5.pth"))
model.eval()

class BinaryThresholdTransform:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        img_np = np.array(img)  # Convert to numpy array
        _, binary_img = cv2.threshold(img_np, self.threshold, 255, cv2.THRESH_BINARY)
        return transforms.functional.to_tensor(binary_img)

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    BinaryThresholdTransform(binary_threshold),
    transforms.Resize((28, 28)),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_character(image):
    """Predict the character of a given image."""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return character_classes[predicted.item()]

def process_images():
    """Load each image and display the sliding window with predictions."""
    for filename in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Could not load image: {filename}")
            continue

        current_x, current_y = 0, 0  # Reset window position for each image
        print(f"Processing image: {filename}")

        while True:
            # Display the image with the sliding window
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(display_image, (current_x, current_y), 
                          (current_x + window_width, current_y + window_height), 
                          (0, 255, 0), 2)
            cv2.imshow("Character Prediction", display_image)

            # Wait for a key press
            key = cv2.waitKey(0)

            if key == ord('k'):
                # Predict the character in the selected window when 'K' is pressed
                cropped_char = image[current_y:current_y+window_height, current_x:current_x+window_width]
                cropped_pil = Image.fromarray(cropped_char)
                predicted_label = predict_character(cropped_pil)
                print(f"Predicted character: {predicted_label}")
            elif key == ord('d'):
                # Move the window to the right
                current_x = min(current_x + step_size, image.shape[1] - window_width)
            elif key == ord('a'):
                # Move the window to the left
                current_x = max(current_x - step_size, 0)
            elif key == ord('s'):
                # Move the window down
                current_y = min(current_y + step_size, image.shape[0] - window_height)
            elif key == ord('w'):
                # Move the window up
                current_y = max(current_y - step_size, 0)
            elif key == 27:  # ESC key to go to the next image
                break

            # If the window is manually closed, stop the script
            if cv2.getWindowProperty("Character Prediction", cv2.WND_PROP_VISIBLE) < 1:
                break

    print("Finished processing all images.")

# Start the process
print("Starting character prediction. Use W/A/S/D to move the window, 'K' to predict the character in the window, and ESC to skip to the next image.")
process_images()

# Cleanup
cv2.destroyAllWindows()

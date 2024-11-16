import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

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
model_path = "character_recognition_model_6.pth"
model = CharacterRecognitionModel(num_classes=14)  # Adjust num_classes as per your character count
model.load_state_dict(torch.load(model_path))
model.eval()

# Define character classes
character_classes = ['dash', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'dot', 'slash', 'space']

# Path to the main folder
main_dir = "split_digits"

# Custom binary threshold transformation
def apply_binary_threshold(image, threshold=200):
    """Apply binary threshold to enhance contrast for character recognition."""
    # Convert PIL image to numpy array
    image_np = np.array(image)
    _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary_image)

# Define the transformation pipeline with the custom binary threshold
transform = transforms.Compose([
    transforms.Lambda(lambda img: apply_binary_threshold(img, threshold=200)),  # Apply binary threshold
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale values
])

def predict_character(image):
    """Predict the character in the given image using the trained model."""
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return character_classes[predicted.item()]

def main():
    # Iterate through each image in the main directory
    for image_name in sorted(os.listdir(main_dir)):
        image_path = os.path.join(main_dir, image_name)
        if os.path.isfile(image_path):
            print(f"\nProcessing image: {image_name}")
            image = Image.open(image_path).convert("L")  # Load image as grayscale
            
            # Predict and display the recognized characters
            predicted_char = predict_character(image)
            print(f"Image: {image_name} -> Predicted character: {predicted_char}")

if __name__ == "__main__":
    main()

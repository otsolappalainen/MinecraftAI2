import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch import nn
from PIL import Image

# Paths
data_dir = r"C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\cropped_digits"
model_path = r"C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\character_recognition_model_7.pth"

# Define character classes
character_classes = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '(', ')', '/', 'space']

# Define a transformation to preprocess images for the model
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Define the character recognition model structure
class CharacterRecognitionModel(nn.Module):
    def __init__(self, num_classes=len(character_classes)):
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

# Load the trained model
model = CharacterRecognitionModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to predict character from image
def predict_character(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return character_classes[predicted.item()]

# Iterate through each labeled folder and identify potential mislabels
outliers = []
for label in os.listdir(data_dir):
    label_folder = os.path.join(data_dir, label)
    if not os.path.isdir(label_folder):
        continue

    print(f"Checking folder '{label}' for outliers...")

    for image_name in os.listdir(label_folder):
        image_path = os.path.join(label_folder, image_name)
        
        # Predict the character
        predicted_label = predict_character(image_path)
        
        # Check if the predicted label matches the folder label
        if label != predicted_label:
            outliers.append((image_path, label, predicted_label))
            print(f"Outlier found: {image_path} | Labeled as: {label}, Predicted as: {predicted_label}")

# Display results
print("\nOutlier Detection Completed")
print(f"Total potential mislabelings found: {len(outliers)}")
for outlier in outliers:
    print(f"Image: {outlier[0]}, Labeled as: {outlier[1]}, Predicted as: {outlier[2]}")

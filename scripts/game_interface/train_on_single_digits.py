import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np

# Parameters
data_dir = "cropped_digits"  # Path to your dataset directory
num_epochs = 100
batch_size = 32
learning_rate = 0.001
model_save_path = "character_recognition_model.pth"
binary_threshold = 200  # Binary threshold cutoff

# Character classes based on folder names
character_classes = sorted(os.listdir(data_dir))

# Function to save the model with a unique name
def save_model_with_unique_name(model, base_path):
    if not os.path.exists(base_path):
        torch.save(model.state_dict(), base_path)
        print(f"Model saved as {base_path}")
    else:
        # Generate a unique filename
        filename, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            new_path = f"{filename}_{counter}{ext}"
            if not os.path.exists(new_path):
                torch.save(model.state_dict(), new_path)
                print(f"Model saved as {new_path}")
                break
            counter += 1

# Custom transform to apply binary threshold
class BinaryThresholdTransform:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img):
        img_np = np.array(img)  # Convert to numpy array
        _, binary_img = cv2.threshold(img_np, self.threshold, 255, cv2.THRESH_BINARY)
        return transforms.functional.to_tensor(binary_img)

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
    BinaryThresholdTransform(binary_threshold),   # Apply binary threshold
    transforms.Resize((28, 28)),                  # Resize to 28x28 pixels
    transforms.Normalize((0.5,), (0.5,))          # Normalize grayscale values
])

# Load the dataset
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model definition - simple CNN
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
        x = self.pool(self.relu(self.conv1(x)))   # Output: [batch_size, 32, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))   # Output: [batch_size, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)                # Flatten for the fully connected layer
        x = self.relu(self.fc1(x))                # Fully connected layer
        x = self.fc2(x)                           # Output layer
        return x

# Initialize model, criterion, and optimizer
num_classes = len(character_classes)
model = CharacterRecognitionModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save the trained model with a unique filename if necessary
save_model_with_unique_name(model, model_save_path)

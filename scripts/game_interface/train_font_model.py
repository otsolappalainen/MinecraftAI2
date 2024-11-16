import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import re

# Define paths for images and labels
IMAGE_DIR = "minecraft_screenshots"
LABEL_FILE = "labeled_coordinates.txt"

# Define the dataset class
class CoordinateDataset(Dataset):
    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir
        self.labels = []
        
        # Load and process the label file
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) < 2:
                    print(f"Skipping badly formatted line: {line.strip()}")
                    continue
                
                image_name = parts[0].strip()
                coords_str = parts[1].strip()

                # Extract coordinates
                try:
                    x, y, z = map(float, coords_str.split())
                    self.labels.append((image_name, x, y, z))
                except ValueError:
                    print(f"Skipping badly formatted row: {line.strip()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.labels[idx][0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))  # Resize for consistent input shape

        # Get coordinates
        x, y, z = self.labels[idx][1:]
        coords = np.array([x, y, z], dtype=np.float32)

        # Convert to PyTorch tensors
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # Normalize and rearrange
        coords = torch.tensor(coords, dtype=torch.float32)
        
        return image, coords

# Define the neural network model
class CoordinateNet(nn.Module):
    def __init__(self):
        super(CoordinateNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 3)  # Predict (x, y, z) coordinates
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset and create dataloaders
train_dataset = CoordinateDataset(IMAGE_DIR, LABEL_FILE)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
model = CoordinateNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop with NaN checks
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, coords in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Check for NaNs in outputs and coordinates
        if torch.isnan(outputs).any() or torch.isnan(coords).any():
            print("NaN detected in outputs or coordinates. Skipping this batch.")
            continue

        loss = criterion(outputs, coords)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    if torch.isnan(torch.tensor(avg_loss)):
        print("NaN detected in average loss, stopping training.")
        break

print("Training completed.")

# Save the trained model
torch.save(model.state_dict(), "coordinate_model.pth")
print("Model saved as coordinate_model.pth")

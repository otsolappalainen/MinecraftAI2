import os
import glob
import pickle
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# Device configuration
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    th.cuda.set_device(0)  # Use GPU 0
    th.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking for speed

# Configuration
MODEL_SAVE_PATH = "models_bc"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
CHUNK_SIZE = 1000  # Number of samples to load at a time
VERBOSE = True

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Define the custom dataset
class ExpertDataset(Dataset):
    def __init__(self, data_directories, chunk_size=1000):
        """
        Dataset for loading expert data in chunks to save memory.

        :param data_directories: List of directories containing expert_data.pkl files.
        :param chunk_size: Number of samples to load into memory at a time.
        """
        self.data_directories = data_directories
        self.chunk_size = chunk_size
        self.files = []
        self.current_chunk = []
        self.current_file_index = 0
        self.current_chunk_start = 0  # Start index of the current chunk

        # Collect all expert_data.pkl files
        for directory in self.data_directories:
            pkl_files = glob.glob(os.path.join(directory, 'expert_data.pkl'))
            self.files.extend(pkl_files)

        if not self.files:
            raise ValueError("No expert_data.pkl files found in the specified directories.")

        # Load the first file and its first chunk
        self._load_file(self.files[self.current_file_index])

    def _load_file(self, file_path):
        """
        Load a file and split it into chunks.
        """
        with open(file_path, 'rb') as f:
            self.current_file_data = pickle.load(f)  # Load the entire file into memory temporarily

        # Reset the chunk index and split the data into chunks
        self.current_chunk_start = 0
        self.current_chunk = self.current_file_data[self.current_chunk_start:self.current_chunk_start + self.chunk_size]

    def _load_next_chunk(self):
        """
        Load the next chunk from the current file or move to the next file.
        """
        self.current_chunk_start += self.chunk_size

        if self.current_chunk_start >= len(self.current_file_data):
            # Move to the next file
            self.current_file_index += 1
            if self.current_file_index >= len(self.files):
                raise StopIteration("No more data to load.")  # End of dataset
            self._load_file(self.files[self.current_file_index])
        else:
            # Load the next chunk within the current file
            self.current_chunk = self.current_file_data[self.current_chunk_start:self.current_chunk_start + self.chunk_size]

    def __len__(self):
        """
        Approximate length of the dataset based on all files.
        """
        total_length = 0
        for file in self.files:
            with open(file, 'rb') as f:
                total_length += len(pickle.load(f))
        return total_length

    def __getitem__(self, idx):
        """
        Fetch the data item at the given index.
        """
        while idx >= len(self.current_chunk):  # Adjust for the current chunk
            self._load_next_chunk()
            idx -= self.chunk_size

        entry = self.current_chunk[idx]
        obs = entry['observation']
        action = entry['action']

        # Validate the existence of 'task' and concatenate
        if 'task' not in obs:
            raise KeyError("'task' key is missing in observation.")
        image = th.tensor(obs['image'], dtype=th.float32)
        other = th.tensor(np.concatenate([obs['other'], obs['task']]), dtype=th.float32)
        action = th.tensor(action, dtype=th.long)

        return {'image': image, 'other': other}, action

# Define the model architecture
class FullModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FullModel, self).__init__()
        # Scalar observation processing
        scalar_input_dim = observation_space['other']
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        scalar_output_size = 128

        # Image processing
        image_input_channels = observation_space['image'][0]  # Should be 3 for RGB images
        self.image_net = nn.Sequential(
            nn.Conv2d(image_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size
        with th.no_grad():
            dummy_input = th.zeros(1, image_input_channels, 224, 224)
            conv_output_size = self.image_net(dummy_input).shape[1]

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(scalar_output_size + conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self._features_dim = 128  # Final feature dimension

        # Output layer
        self.action_head = nn.Linear(self._features_dim, action_space)

    def forward(self, observations):
        scalar_features = self.scalar_net(observations['other'])
        image_features = self.image_net(observations['image'])
        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        action_logits = self.action_head(fused_features)
        return action_logits

def main():
    # Specify the directories containing your expert data
    data_directories = [
        r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\expert_data\session_*"
    ]

    # Create dataset and dataloader
    dataset = ExpertDataset(data_directories)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Define observation and action spaces based on your data
    first_sample = dataset[0]
    observation_space = {
        'image': (3, 224, 224),
        'other': first_sample[0]['other'].shape[0]  # Dynamically fetch size
    }
    action_space = 18  # Number of actions

    # Initialize the model
    model = FullModel(observation_space, action_space).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for batch in pbar:
            observations, actions = batch
            # Move data to device
            observations = {k: v.to(device) for k, v in observations.items()}
            actions = actions.to(device)

            optimizer.zero_grad()

            # Forward pass
            action_logits = model(observations)
            loss = criterion(action_logits, actions)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item() * actions.size(0)
            _, predicted = th.max(action_logits.data, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

            pbar.set_postfix({'loss': epoch_loss / total, 'acc': 100. * correct / total})

        # Save model checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pt")
        th.save(model.state_dict(), checkpoint_path)
        if VERBOSE:
            print(f"Epoch {epoch+1} completed. Model saved to '{checkpoint_path}'.")

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, "final_model.pt")
    th.save(model.state_dict(), final_model_path)
    if VERBOSE:
        print(f"Training completed. Final model saved to '{final_model_path}'.")

if __name__ == "__main__":
    main()

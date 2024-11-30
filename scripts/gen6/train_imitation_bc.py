import os
import glob
import pickle
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

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
LOG_LEVEL = logging.DEBUG  # Change this to logging.INFO for less detailed logs

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Define the custom dataset
class ExpertDataset(Dataset):
    def __init__(self, data_directories, chunk_size=1000):
        self.data_directories = data_directories
        self.chunk_size = chunk_size
        self.files = []
        self.data = []  # Store all data in memory
        
        # Collect all expert_data.pkl files
        for directory in self.data_directories:
            pkl_files = glob.glob(os.path.join(directory, 'expert_data.pkl'))
            self.files.extend(pkl_files)

        if not self.files:
            raise ValueError("No expert_data.pkl files found in the specified directories.")
        
        # Load all files into memory
        for file_path in self.files:
            logger.info(f"Loading file: {file_path}")
            with open(file_path, 'rb') as f:
                file_data = pickle.load(f)
                self.data.extend(file_data)
                logger.info(f"Loaded {len(file_data)} samples from file")
        
        logger.info(f"Total samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            entry = self.data[idx]
            
            obs = entry['observation']
            action = entry['action']

            if 'task' not in obs:
                raise KeyError(f"'task' key missing in observation for index {idx}")
            
            image = th.tensor(obs['image'], dtype=th.float32)
            other = th.tensor(np.concatenate([obs['other'], obs['task']]), dtype=th.float32)
            action = th.tensor(action, dtype=th.long)

            # Verify tensor shapes
            logger.debug(f"Getting item {idx}: Image shape: {image.shape}, Other shape: {other.shape}, Action: {action.item()}")
            
            return {'image': image, 'other': other}, action

        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            raise


# Define the model architecture
class FullModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FullModel, self).__init__()
        
        # Image processing network
        image_input_channels = observation_space['image'][0]
        self.image_net = nn.Sequential(
            nn.Conv2d(image_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device)  # Move to device immediately

        # Compute CNN output size
        with th.no_grad():
            dummy_input = th.zeros(1, image_input_channels, 224, 224).to(device)
            conv_output_size = self.image_net(dummy_input).shape[1]

        # Scalar observation processing
        scalar_input_dim = observation_space['other']
        scalar_output_size = 128
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ).to(device)  # Move to device immediately

        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(scalar_output_size + conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        ).to(device)  # Move to device immediately

        self._features_dim = 128
        self.action_head = nn.Linear(self._features_dim, action_space).to(device)

    def forward(self, observations):
        scalar_features = self.scalar_net(observations['other'])
        image_features = self.image_net(observations['image'])
        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        action_logits = self.action_head(fused_features)
        return action_logits




def main():
    data_directories = [
        r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\expert_data\session_*"
    ]
    
    logger.info("Initializing dataset and dataloader")
    dataset = ExpertDataset(data_directories)
    
    # Test loading a few samples directly from dataset
    logger.info("Testing direct dataset access...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        logger.info(f"Sample {i}: Action = {sample[1]}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Changed to 0 for debugging
        pin_memory=True
    )
    
    logger.info(f"Dataset length: {len(dataset)}")
    
    # Test the dataloader directly
    logger.info("Testing dataloader...")
    test_batch = next(iter(dataloader))
    logger.info(f"Test batch shapes: {[t.shape for t in test_batch[0].values()]}, {test_batch[1].shape}")
    
    first_sample = dataset[0]
    observation_space = {
        'image': (3, 224, 224),
        'other': first_sample[0]['other'].shape[0]
    }
    action_space = 18

    model = FullModel(observation_space, action_space).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        logger.info(f"Starting epoch {epoch+1}/{EPOCHS}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        try:
            for batch_idx, batch in enumerate(pbar):
                observations, actions = batch
                
                # Debug info
                logger.debug(f"Batch {batch_idx}: observations shape - image: {observations['image'].shape}, "
                           f"other: {observations['other'].shape}, actions: {actions.shape}")
                
                # Move data to device
                observations = {k: v.to(device) for k, v in observations.items()}
                actions = actions.to(device)

                optimizer.zero_grad()
                action_logits = model(observations)
                
                # Debug prediction shapes
                logger.debug(f"Logits shape: {action_logits.shape}, Actions shape: {actions.shape}")
                
                loss = criterion(action_logits, actions)
                loss.backward()
                optimizer.step()

                # Update statistics
                batch_size = actions.size(0)
                epoch_loss += loss.item() * batch_size
                _, predicted = th.max(action_logits.data, 1)
                total += batch_size
                correct += (predicted == actions).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                'acc': f'{(correct/total)*100:.2f}%' if total > 0 else 'N/A'})

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise e

        if total == 0:
            logger.error("No samples were processed in this epoch!")
            continue

        avg_loss = epoch_loss / total
        accuracy = correct / total
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        if (epoch + 1) % 5 == 0:
            model_save_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth")
            th.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()

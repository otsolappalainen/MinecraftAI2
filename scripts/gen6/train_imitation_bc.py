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
LOG_LEVEL = logging.INFO # Change this to logging.INFO for less detailed logs

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Define the custom dataset
class ExpertDataset(Dataset):
    def __init__(self, data_directories, chunk_size=1000):
        self.data_directories = data_directories
        self.files = []
        self.data = []
        
        # Collect all expert_data.pkl files
        for directory in self.data_directories:
            pkl_files = glob.glob(os.path.join(directory, 'expert_data.pkl'))
            self.files.extend(pkl_files)

        if not self.files:
            raise ValueError("No expert_data.pkl files found")
        
        # Load all files
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

            # Create normalized observations
            image = th.tensor(obs['image'], dtype=th.float32)
            other = obs['other'].copy()
            
            # Normalize yaw
            yaw_idx = 0
            yaw = other[yaw_idx] 
            normalized_yaw = ((yaw + 180) % 360) - 180
            other[yaw_idx] = normalized_yaw / 180.0

            other = th.tensor(other, dtype=th.float32)
            
            # Create default task vector [0,1,0,0,...] 
            task = th.zeros(20, dtype=th.float32)
            task[1] = 1.0

            return {
                'image': image,
                'other': other,
                'task': task
            }, th.tensor(action, dtype=th.long)

        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            raise


def normalize_observations(obs):
    """Normalize observation values"""
    normalized = {}
    
    # Image already normalized to [0,1]
    normalized['image'] = obs['image']
    
    # Create copy of other observations
    normalized['other'] = obs['other'].copy()
    
    # Normalize yaw to [-180, 180] then to [-1, 1]
    yaw_idx = 0  # First value in 'other' array
    yaw = normalized['other'][yaw_idx]
    normalized_yaw = ((yaw + 180) % 360) - 180  # Wrap to [-180, 180]
    normalized['other'][yaw_idx] = normalized_yaw / 180.0  # Scale to [-1, 1]
    
    return normalized


# Define the model architecture
class FullModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FullModel, self).__init__()
        
        # Calculate correct input dimension (other + task)
        scalar_input_dim = observation_space['other'] + 20  # 8 other features + 20 task features
        logger.info(f"Scalar input dim: {scalar_input_dim}")
        
        # Scalar observation processing with dropout
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        scalar_output_size = 128

        # Image processing with batch normalization
        image_input_channels = observation_space['image'][0]
        self.image_net = nn.Sequential(
            nn.Conv2d(image_input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Flatten()
        )
        
        # Compute CNN output size
        with th.no_grad():
            dummy_input = th.zeros(1, image_input_channels, 224, 224)
            conv_output_size = self.image_net(dummy_input).shape[1]

        # Add fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(scalar_output_size + conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
        self._features_dim = 128
        
        # Action head with lower initial weights
        self.action_head = nn.Linear(self._features_dim, action_space)
        nn.init.normal_(self.action_head.weight, mean=0.0, std=0.01)

    def forward(self, observations, eval_mode=False):
        if not eval_mode:
            self.train()
        else:
            self.eval()
        
        # Debug shapes
        logger.debug(f"Other shape: {observations['other'].shape}")
        logger.debug(f"Task shape: {observations['task'].shape}")
        
        other_combined = th.cat([observations['other'], observations['task']], dim=1)
        logger.debug(f"Combined input shape: {other_combined.shape}")
        
        scalar_features = self.scalar_net(other_combined)
        image_features = self.image_net(observations['image'])
        
        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        
        if not eval_mode:
            noise = th.randn_like(fused_features) * 0.1
            fused_features = fused_features + noise
            
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

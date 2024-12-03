# File: automatic_data_generator.py

import torch as th
import torch.nn as nn
import numpy as np
import asyncio
import time
import os
import pickle
import logging
from collections import deque
import websockets
from datetime import datetime

from mod_env_v1 import MinecraftEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Configuration
DATA_DIR = r'E:\automatic_model_tool_spam'  # Change data directory
BUFFER_SIZE = 5  # Steps to save before/after block break
EPISODES = 100
STEPS_PER_EPISODE = 1000
VALID_Y_RANGE = (132, 133)

# Create base directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

class FullModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(FullModel, self).__init__()
        
        # Calculate input dim including task vector (29 other + 20 task)
        scalar_input_dim = observation_space['other'] + 20  # Should be 49
        logger.info(f"Scalar input dim: {scalar_input_dim}")
        
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_input_dim, 256),  # Increased first layer
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
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

    def forward(self, observations, eval_mode=True):
        if not eval_mode:
            self.train()
        else:
            self.eval()
            
        other_combined = th.cat([observations['other'], observations['task']], dim=1)
        scalar_features = self.scalar_net(other_combined)
        image_features = self.image_net(observations['image'])
        
        combined_input = th.cat([scalar_features, image_features], dim=1)
        fused_features = self.fusion_layers(combined_input)
        
        if eval_mode:
            noise = th.randn_like(fused_features) * 0.1
            fused_features = fused_features + noise
            
        action_logits = self.action_head(fused_features)
        return action_logits

class DataCollector:
    def __init__(self, model_path):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.buffer = deque(maxlen=BUFFER_SIZE * 2 + 1)
        self.env = None
        os.makedirs(DATA_DIR, exist_ok=True)
        self.running = True

    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.env:
            try:
                await self.env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
            self.env = None
        
        # Force cleanup of pending tasks
        try:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error cleaning up tasks: {e}")

    def load_model(self, model_path):
        observation_space = {
            'image': (3, 224, 224),
            'other': 29  # Keep at 29 to match trained model
        }
        action_space = 18
        model = FullModel(observation_space, action_space).to(self.device)
        model.load_state_dict(th.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    async def collect_data(self):
        try:
            for episode in range(EPISODES):
                if not self.running:
                    break
                    
                logger.info(f"Starting episode {episode+1}")
                episode_data = []
                self.buffer.clear()
                
                try:
                    # Create and initialize environment
                    self.env = MinecraftEnv()
                    
                    # Wait for initial connection with retry
                    retry_count = 0
                    max_retries = 3
                    while retry_count < max_retries:
                        timeout = 5.0
                        start_time = time.time()
                        while not self.env.connected and (time.time() - start_time) < timeout:
                            await asyncio.sleep(0.1)
                        
                        if self.env.connected:
                            break
                            
                        retry_count += 1
                        logger.warning(f"Connection attempt {retry_count} failed, retrying...")
                        await asyncio.sleep(1.0)
                    
                    if not self.env.connected:
                        logger.error("Failed to connect after retries")
                        continue
                    
                    # Initialize task
                    task = th.zeros(20, dtype=th.float32)
                    task[1] = 1.0
                    
                    # Single reset and wait for result
                    logger.info("Resetting environment...")
                    obs, info = await self.env.reset()
                    
                    if obs is None:
                        logger.error("Failed to get initial observation")
                        continue
                    
                    # Validate initial state
                    if not self.validate_observation(obs):
                        logger.error("Invalid initial observation")
                        continue
                    
                    # Run episode
                    for step in range(STEPS_PER_EPISODE):
                        try:
                            step_start_time = time.time()
                            # Process step
                            norm_obs = self.normalize_observation(obs)
                            action = self.get_model_action(norm_obs, task)
                            
                            # Take action and wait for result
                            new_obs, reward, done, truncated, info = await self.env.step(action)
                            
                            # Only continue if we got valid observation
                            if not self.validate_observation(new_obs):
                                logger.error(f"Invalid observation at step {step}")
                                await asyncio.sleep(0.1)  # Short delay on error
                                continue  # Skip this step rather than break
                                
                            # Update observation
                            obs = new_obs
                            
                            # Store in buffer
                            self.buffer.append({
                                'observation': norm_obs,
                                'action': action,
                                'next_observation': None,
                                'done': False,
                                'info': {}
                            })
                            
                            # Process block breaks...
                            # Extract block data from other array (indices 8 onwards are block data)
                            blocks_data = obs['other'][8:]  # Get block data portion
                            if np.any(blocks_data):  # Only print if any non-zero values exist
                                print(f"broken blocks this step: {blocks_data}")
                            
                            # Reshape into blocks (5 blocks × 4 features)
                            blocks = blocks_data.reshape(-1, 4)  # Reshape into [num_blocks, 4] array
                            
                            # Filter valid blocks (non-zero blocks at valid Y coordinates)
                            valid_breaks = []
                            for block in blocks:
                                if any(block):  # If any value in block is non-zero
                                    blocktype, x, y, z = block
                                    if VALID_Y_RANGE[0] <= y <= VALID_Y_RANGE[1]:  # Denormalize y coordinate
                                        valid_breaks.append({
                                            'blocktype': blocktype,  # Denormalize
                                            'blockx': x,
                                            'blocky': y,
                                            'blockz': z
                                        })
                            
                            if valid_breaks:  # Only print if there are valid breaks
                                print(f"valid broken blocks this step: {valid_breaks}")
                            
                            if valid_breaks:
                                episode_data.extend(list(self.buffer))
                                self.buffer.clear()
                            
                            if done:
                                break
                            
                            # Dynamic sleep based on processing time
                            processing_time = time.time() - step_start_time
                            if processing_time < 0.1:  # Target 70ms minimum between actions
                                await asyncio.sleep(0.1 - processing_time)
                                
                        except Exception as e:
                            logger.error(f"Error in step {step}: {e}")
                            await asyncio.sleep(0.1)  # Recovery delay
                            continue
                
                    # Save episode data
                    if episode_data:
                        # Create timestamp folder name
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        session_folder = f'session_{timestamp}'
                        
                        # Create full save path
                        save_path = os.path.join(DATA_DIR, session_folder, f'episode_{episode}', 'expert_data.pkl')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        
                        with open(save_path, 'wb') as f:
                            pickle.dump(episode_data, f)
                        logger.info(f"Saved {len(episode_data)} samples to {save_path}")

                except websockets.ConnectionClosed:
                    logger.warning("WebSocket connection closed, ending episode")
                    continue
                except Exception as e:
                    logger.error(f"Error in episode {episode}: {str(e)}")
                finally:
                    if self.env:
                        try:
                            await self.env.close()
                        except Exception as e:
                            logger.error(f"Error closing environment: {e}")
                        self.env = None
                    await asyncio.sleep(1.0)
        finally:
            await self.cleanup()

    def validate_observation(self, obs):
        """Validate observation structure and values"""
        if obs is None:
            return False
            
        required_keys = ['image', 'other', 'task']
        return all(k in obs for k in required_keys)

    def convert_to_model_format(self, obs):
        """Convert normalized 48-dim to 29-dim format (no renormalization)"""
        # Get already normalized data
        basic_data = obs['other'][:9]  # Include alive flag
        blocks_data = obs['other'][9:]
        
        # Count number of broken blocks
        num_blocks = len(blocks_data) // 4
        blocks_broken = sum(1 for i in range(num_blocks) 
                           if any(blocks_data[i*4:(i+1)*4]))
        
        # Replace alive flag with block count
        model_obs = basic_data.copy()
        model_obs[8] = blocks_broken / 5.0  # normalize block count
        
        # Pad with zeros to match 29-dim format (9 basic + 20 padding)
        padded_other = np.zeros(29, dtype=np.float32)
        padded_other[:9] = model_obs  # First 9 dimensions
        
        return {
            'image': obs['image'],
            'other': padded_other,  # 29-dim vector (9 basic + 20 zeros)
            'task': obs['task']
        }

    def normalize_observation(self, obs):
        """Store full 48-dim normalized data"""
        # Basic observations 
        yaw = obs['other'][6]
        
        # Get raw data
        basic_data = obs['other'][:8]
        blocks_data = obs['other'][8:]
        
        # Create normalized basic state
        basic_obs = np.array([
            basic_data[0] / 20000.0,  # x
            basic_data[1] / 20000.0,  # y 
            basic_data[2] / 256.0,    # z
            np.sin(np.deg2rad(yaw)),  # sin_yaw
            np.cos(np.deg2rad(yaw)),  # cos_yaw
            basic_data[3] / 20.0,     # health
            basic_data[4] / 20.0,     # hunger
            basic_data[5] / 180.0,    # pitch
            1.0                       # alive
        ], dtype=np.float32)
        
        # Normalize block data (20 values)
        max_blocks = 5
        block_features = 4
        normalized_blocks = np.zeros(max_blocks * block_features, dtype=np.float32)
        
        if len(blocks_data) > 0:
            for i in range(max_blocks):
                base_idx = i * block_features
                if base_idx + block_features <= len(blocks_data):
                    normalized_blocks[base_idx] = blocks_data[base_idx] / 10.0
                    normalized_blocks[base_idx + 1] = blocks_data[base_idx + 1] / 20000.0
                    normalized_blocks[base_idx + 2] = blocks_data[base_idx + 2] / 256.0
                    normalized_blocks[base_idx + 3] = blocks_data[base_idx + 3] / 20000.0

        return {
            'image': obs['image'].transpose(2, 0, 1).astype(np.float32),
            'other': np.concatenate([basic_obs, normalized_blocks]),
            'task': obs['task']
        }

    def get_model_action(self, obs, task):
        """Get action using model format"""
        with th.no_grad():
            # Convert to model format (29-dim)
            model_obs = self.convert_to_model_format(obs)
            
            # Ensure dimensions match trained model
            image = th.FloatTensor(model_obs['image']).unsqueeze(0)
            other = th.FloatTensor(model_obs['other']).unsqueeze(0)  # Should be 29-dim
            task = th.FloatTensor(task).unsqueeze(0)  # 20-dim
            
            # Verify shapes
            assert other.shape[1] == 29, f"Other shape should be [1,29], got {other.shape}"
            assert task.shape[1] == 20, f"Task shape should be [1,20], got {task.shape}"
            
            observations = {
                'image': image.to(self.device),
                'other': other.to(self.device),
                'task': task.to(self.device)
            }
            
            logits = self.model(observations)
            temperature = 2.0
            boost_factor = 0.3
            
            action_probs = th.softmax(logits / temperature, dim=1)
            action_probs = (1 - boost_factor) * action_probs + boost_factor / action_probs.shape[1]
            
            return th.distributions.Categorical(action_probs).sample().item()

if __name__ == "__main__":
    try:
        MODEL_PATH = r"C:\Users\odezz\source\MinecraftAI2\scripts\gen6\models_bc\model_epoch_10.pth"
        collector = DataCollector(MODEL_PATH)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(collector.collect_data())
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            collector.running = False
        finally:
            # Ensure proper cleanup
            loop.run_until_complete(collector.cleanup())
            loop.close()
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Data collection complete")
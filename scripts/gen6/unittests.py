import sys
import os
import unittest
import torch as th
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from train_dqn_from_bc import BCMatchingFeatureExtractor

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBCMatchingFeatureExtractor(unittest.TestCase):
  def setUp(self):
    """Setup test fixtures"""
    # Define observation space similar to the actual environment
    self.observation_space = Dict({
      'image': Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
      'other': Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32),
      'task': Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    })

    # Create dummy inputs
    self.batch_size = 2
    self.dummy_image = th.randn(self.batch_size, 3, 224, 224)
    self.dummy_other = th.randn(self.batch_size, 28)
    self.dummy_task = th.randn(self.batch_size, 20)
    
    # Initialize feature extractor
    self.feature_extractor = BCMatchingFeatureExtractor(
      observation_space=self.observation_space,
      features_dim=128
    )

  def test_initialization(self):
    """Test proper initialization of the feature extractor"""
    self.assertEqual(self.feature_extractor._features_dim, 128)
    self.assertIsInstance(self.feature_extractor.scalar_net, th.nn.Sequential)
    self.assertIsInstance(self.feature_extractor.image_net, th.nn.Sequential)
    self.assertIsInstance(self.feature_extractor.fusion_layers, th.nn.Sequential)

  def test_forward_pass_shape(self):
    """Test forward pass returns correct output shape"""
    observations = {
      'image': self.dummy_image,
      'other': self.dummy_other,
      'task': self.dummy_task
    }
    
    with th.no_grad():
      output = self.feature_extractor.forward(observations)
    
    expected_shape = (self.batch_size, 128)
    self.assertEqual(output.shape, expected_shape)

  def test_scalar_network(self):
    """Test scalar network processes inputs correctly"""
    combined = th.cat([self.dummy_other, self.dummy_task], dim=1)
    
    with th.no_grad():
      output = self.feature_extractor.scalar_net(combined)
    
    expected_shape = (self.batch_size, 128)
    self.assertEqual(output.shape, expected_shape)

  def test_image_network(self):
    """Test image network processes inputs correctly"""
    with th.no_grad():
      output = self.feature_extractor.image_net(self.dummy_image)
    
    # The output should be flattened
    self.assertEqual(len(output.shape), 2)
    self.assertEqual(output.shape[0], self.batch_size)

  def test_input_validation(self):
    """Test handling of invalid inputs"""
    # Test missing key
    invalid_observations = {
      'image': self.dummy_image,
      'other': self.dummy_other
      # Missing 'task' key
    }
    
    with self.assertRaises(Exception):
      self.feature_extractor.forward(invalid_observations)

  def test_dtype_consistency(self):
    """Test handling of different input dtypes"""
    # Create observations with float32 dtype
    observations_float32 = {
      'image': self.dummy_image.float(),
      'other': self.dummy_other.float(),
      'task': self.dummy_task.float()
    }
    
    with th.no_grad():
      output = self.feature_extractor.forward(observations_float32)
    
    self.assertEqual(output.dtype, th.float32)

  def test_device_consistency(self):
    """Test device handling"""
    if th.cuda.is_available():
      device = th.device('cuda')
      feature_extractor = self.feature_extractor.to(device)
      
      observations = {
        'image': self.dummy_image.to(device),
        'other': self.dummy_other.to(device),
        'task': self.dummy_task.to(device)
      }
      
      with th.no_grad():
        output = feature_extractor.forward(observations)
      
      # Change from direct device comparison to checking if both are on CUDA
      self.assertTrue(output.is_cuda)

  def test_gradient_flow(self):
    """Test gradient flow through the network"""
    observations = {
      'image': self.dummy_image,
      'other': self.dummy_other,
      'task': self.dummy_task
    }
    
    output = self.feature_extractor.forward(observations)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist
    for param in self.feature_extractor.parameters():
      self.assertIsNotNone(param.grad)

if __name__ == '__main__':
  unittest.main()
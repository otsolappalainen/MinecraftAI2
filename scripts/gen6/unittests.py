import unittest
import numpy as np
import torch as th
import gymnasium as gym

from train_dqn_from_bc import BCMatchingFeatureExtractor

class TestBCMatchingFeatureExtractor(unittest.TestCase):
  def setUp(self):
    # Create a mock observation space matching the real environment
    self.observation_space = gym.spaces.Dict({
      'image': gym.spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32),
      'other': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32),
      'task': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    })

  def test_feature_extractor_initialization(self):
    try:
      extractor = BCMatchingFeatureExtractor(self.observation_space)
      self.assertIsNotNone(extractor)
      self.assertEqual(extractor._features_dim, 128)
    except Exception as e:
      self.fail(f"Feature extractor initialization failed: {str(e)}")

  def test_feature_extractor_forward_pass(self):
    extractor = BCMatchingFeatureExtractor(self.observation_space)
    
    # Create dummy input matching the observation space
    dummy_obs = {
      'image': th.zeros((1, 3, 224, 224), dtype=th.float32),
      'other': th.zeros((1, 28), dtype=th.float32),
      'task': th.zeros((1, 20), dtype=th.float32)
    }

    # Test forward pass
    with th.no_grad():
      output = extractor.forward(dummy_obs)
      
    # Check output shape
    self.assertEqual(output.shape, (1, 128))

  def test_dropout_layers(self):
    extractor = BCMatchingFeatureExtractor(self.observation_space)
    
    # Verify Dropout2d layer configuration
    dropout2d_found = False
    for layer in extractor.image_net:
      if isinstance(layer, th.nn.Dropout2d):
        dropout2d_found = True
        self.assertEqual(layer.p, 0.3)
    
    self.assertTrue(dropout2d_found, "Dropout2d layer not found in image_net")

  def test_training_mode_effect(self):
    extractor = BCMatchingFeatureExtractor(self.observation_space)
    dummy_obs = {
      'image': th.zeros((1, 3, 224, 224), dtype=th.float32),
      'other': th.zeros((1, 28), dtype=th.float32),
      'task': th.zeros((1, 20), dtype=th.float32)
    }

    # Test in eval mode
    extractor.eval()
    with th.no_grad():
      eval_output = extractor.forward(dummy_obs)

    # Test in train mode
    extractor.train()
    with th.no_grad():
      train_output = extractor.forward(dummy_obs)

    # Outputs should be different in train mode due to dropout
    self.assertFalse(th.allclose(eval_output, train_output))

  def test_feature_extraction_no_nan(self):
    extractor = BCMatchingFeatureExtractor(self.observation_space)
    dummy_obs = {
      'image': th.rand((1, 3, 224, 224), dtype=th.float32),
      'other': th.rand((1, 28), dtype=th.float32),
      'task': th.rand((1, 20), dtype=th.float32)
    }

    with th.no_grad():
      output = extractor.forward(dummy_obs)
      
    # Check for NaN values
    self.assertFalse(th.isnan(output).any())

if __name__ == '__main__':
  unittest.main()
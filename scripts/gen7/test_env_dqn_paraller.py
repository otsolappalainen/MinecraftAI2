import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from env_dqn_paraller import MinecraftEnv
import json

class TestMinecraftEnv(unittest.TestCase):

    def setUp(self):
        self.env = MinecraftEnv()
        def test_observation_space(self):
          """Test observation space structure and shapes"""
          self.assertEqual(len(self.env.observation_space.spaces), 6)
          self.assertEqual(self.env.observation_space['image'].shape, (3, 120, 120))
          self.assertEqual(self.env.observation_space['tasks'].shape, (10,))
          self.assertEqual(self.env.observation_space['blocks'].shape, (4,))
          self.assertEqual(self.env.observation_space['hand'].shape, (5,))
          self.assertEqual(self.env.observation_space['target_block'].shape, (1,))
          self.assertEqual(self.env.observation_space['flattened_matrix'].shape, (729,))
          
        def test_action_mapping(self):
          """Test action mapping completeness and validity"""
          self.assertEqual(len(self.env.ACTION_MAPPING), 18)
          self.assertEqual(self.env.ACTION_MAPPING[0], "move_forward")
          self.assertEqual(self.env.ACTION_MAPPING[17], "no_op")

        def test_normalize_blocks(self):
          """Test blocks normalization"""
          # Test empty list
          result = self.env.normalize_blocks([])
          self.assertEqual(result.shape, (4,))
          self.assertTrue(np.all(result == 0))

          # Test single block
          block = {'blocktype': 1, 'blockx': 2, 'blocky': 3, 'blockz': 4}
          result = self.env.normalize_blocks([block])
          self.assertEqual(result.shape, (4,))
          self.assertEqual(result[0], 1.0)
          
        def test_normalize_task(self):
          """Test task normalization"""
          result = self.env.normalize_task({})
          self.assertEqual(result.shape, (10,))
          self.assertEqual(result[2], self.env.normalize_value(-63, *self.env.height_range))
          
        def test_get_default_state(self):
          """Test default state generation"""
          state = self.env._get_default_state()
          self.assertIsInstance(state, dict)
          self.assertEqual(state['image'].shape, (3, 120, 120))
          self.assertTrue(np.all(state['image'] == 0))
          self.assertEqual(len(state['tasks']), 10)
          
        @patch('mss.mss')
        def test_capture_screenshot_error(self, mock_mss):
          """Test screenshot capture error handling"""
          mock_mss.side_effect = Exception("Screenshot error")
          result = self.env.capture_screenshot(0)
          self.assertEqual(result.shape, (3, 120, 120))
          self.assertTrue(np.all(result == 0))

        def test_normalize_hand(self):
          """Test hand state normalization"""
          # Test default values
          result = self.env.normalize_hand({})
          self.assertEqual(result.shape, (5,))
          self.assertTrue(np.all(result == 0))
          
          # Test with data
          hand_data = {'held_item': [1,2,3,4,5]}
          result = self.env.normalize_hand(hand_data)
          self.assertEqual(result.shape, (5,))
          self.assertEqual(result[0], 1)

if __name__ == '__main__':
    unittest.main()
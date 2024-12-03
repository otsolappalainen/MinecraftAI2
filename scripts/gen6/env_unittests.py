# test_minecraft_env.py
import unittest
from unittest.mock import patch, AsyncMock
import asyncio
import numpy as np
from dqn_env import MinecraftEnv
import json

class TestMinecraftEnv(unittest.TestCase):

    @patch('dqn_env.websockets.connect', new_callable=AsyncMock)
    def setUp(self, mock_websocket_connect):
        self.env = MinecraftEnv()
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.env.connect())
        self.mock_websocket = mock_websocket_connect.return_value
        self.mock_websocket.recv = AsyncMock()

    def tearDown(self):
        self.env.close()

    def test_initialization(self):
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env.cumulative_reward, 0.0)
        self.assertTrue(self.env.connected)

    def test_step(self):
        self.mock_websocket.recv.return_value = json.dumps({
            'broken_blocks': [],
            'alive': True,
            'x': -16.0,
            'y': -63.0,
            'health': 20.0,
            'z': -106.0,
            'pitch': 0.0,
            'inventory': {str(i): 'Item' for i in range(36)},
            'yaw': 0.0,
            'hunger': 20
        })

        action = 0  # move_forward
        state, reward, done, _, info = self.env.step(action)
        self.assertIsInstance(state, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_reset(self):
        self.mock_websocket.recv.return_value = json.dumps({
            'broken_blocks': [],
            'alive': True,
            'x': -16.0,
            'y': -63.0,
            'health': 20.0,
            'z': -106.0,
            'pitch': 0.0,
            'inventory': {str(i): 'Item' for i in range(36)},
            'yaw': 0.0,
            'hunger': 20
        })

        state, info = self.env.reset()
        self.assertIsInstance(state, dict)
        self.assertIsInstance(info, dict)

    def test_step_timeout(self):
        self.mock_websocket.recv.side_effect = asyncio.TimeoutError

        action = 0  # move_forward
        state, reward, done, _, info = self.env.step(action)
        self.assertIsNone(state)
        self.assertEqual(reward, self.env.step_penalty)
        self.assertFalse(done)
        self.assertIsInstance(info, dict)

    def test_reset_timeout(self):
        self.mock_websocket.recv.side_effect = asyncio.TimeoutError

        state, info = self.env.reset()
        self.assertIsNone(state)
        self.assertIsInstance(info, dict)

if __name__ == '__main__':
    unittest.main()
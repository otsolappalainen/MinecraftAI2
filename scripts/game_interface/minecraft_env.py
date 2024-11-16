import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from control_player import MinecraftAgent
import time

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
RESET = "\033[0m"



class MinecraftEnv(gym.Env):
    def __init__(self, action_delay=0.1, max_episode_length=1000):
        super(MinecraftEnv, self).__init__()
        self.agent = MinecraftAgent()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([-100000, -100000, -180, -90, -100000, -100000]),
            high=np.array([100000, 100000, 180, 90, 100000, 100000]),
            dtype=np.float32
        )

        self.action_map = {
            0: 'move_forward',
            1: 'move_backward',
            2: 'strafe_left',
            3: 'strafe_right',
            4: 'turn_left',
            5: 'turn_right'
        }

        self.max_episode_length = max_episode_length
        self.step_counter = 0
        self.action_delay = action_delay
        self.reached_target = False
        self.target_x, self.target_z = 0.0, 0.0  # Target coordinates
        self.prev_distance_to_target = None

    def step(self, action):
        # Ensure action is correctly processed
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = action.item()  # Extract single-element as scalar
            else:
                raise ValueError(f"Action array has more than one element, which is unexpected: {action}")

        if not isinstance(action, int):
            raise ValueError(f"Action must be an integer, but got type {type(action)}")

        # Check if the agent has already reached the target
        if self.reached_target:
            x, z, yaw, pitch = self.agent.state if self.agent.state else (0, 0, 0, 0)
            obs = np.array([x, z, yaw, pitch, self.target_x, self.target_z], dtype=np.float32)
            return obs, 0, True, False, {}

        # Ensure action is within valid bounds
        if action < 0 or action >= len(self.action_map):
            raise ValueError(f"Action index {action} is out of bounds for defined actions.")

        # Map the integer action to its corresponding string using `action_map`
        action_str = self.action_map[action]
        
        # Perform the action
        move_start_time = time.time()
        self.agent.perform_action(action_str)
        elapsed_time = time.time() - move_start_time
        print(f"Action '{action_str}' performed in {elapsed_time:.2f} seconds")
        time.sleep(self.action_delay)

        # Update agent's state
        self.agent.get_state()

        # Retrieve new state
        if self.agent.state:
            x, z, yaw, pitch = self.agent.state
        else:
            x, z, yaw, pitch = 0, 0, 0, 0  # Fallback if state retrieval fails

        # Calculate distance to the target
        new_distance_to_target = np.sqrt((x - self.target_x) ** 2 + (z - self.target_z) ** 2)
        old_distance_to_target = self.prev_distance_to_target or new_distance_to_target
        self.prev_distance_to_target = new_distance_to_target

        # Reward calculation based on the new state
        reward = (old_distance_to_target - new_distance_to_target) - 0.3  # Small step penalty

        # Check if agent reached the target
        done = new_distance_to_target < 10
        truncated = False
        if done:
            reward += 1000  # Large reward for reaching the target
            self.reached_target = True
        elif self.step_counter >= self.max_episode_length:
            done = True
            truncated = True

        # Create observation array
        obs = np.array([x, z, yaw, pitch, self.target_x, self.target_z], dtype=np.float32)

        # Check for any invalid values in obs
        if len(obs) != 6 or not np.all(np.isfinite(obs)):
            print("Invalid observation detected, resetting to default values.")
            obs = np.zeros(6, dtype=np.float32)

        # Log debugging information
        print(
            f"Action: {action_str} | Coordinates: X={x:.2f}, Z={z:.2f} | "
            f"Yaw: {yaw:.2f} | Pitch: {pitch:.2f} | Reward: {reward:.2f} | "
            f"Done: {done} | Truncated: {truncated}"
        )

        self.step_counter += 1
        return obs, reward, done, truncated, {}
    



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set the target position to fixed coordinates
        self.target_x = 100  # Adjust this value as needed
        self.target_z = -140  # Adjust this value as needed

        # Reset agent's state
        self.agent.get_state()
        self.step_counter = 0
        self.prev_distance_to_target = None

        if self.agent.state:
            x, z, yaw, pitch = self.agent.state
            print(
                f"Starting Position: X={x:.2f}, Z={z:.2f}, Yaw={yaw:.2f}, Pitch={pitch:.2f} | "
                f"Target: X={self.target_x}, Z={self.target_z}"
            )
            return np.array([x, z, yaw, pitch, self.target_x, self.target_z], dtype=np.float32), {}
        else:
            return np.zeros(6, dtype=np.float32), {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass






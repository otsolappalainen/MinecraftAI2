import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from agent import MinecraftAgent
from colorama import Fore, Style
import cv2
import os
import datetime
import torch
import sys

# -------------------- Configuration --------------------

# Constants
ACTION_DELAY = 0.2
MAX_EPISODE_LENGTH = 1000
TASK_SIZE = 20
REWARD_MODEL = "constant_x"  # or "random_task"
TRAINING_LOGS_DIR = "training_logs"
os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)

# -------------------- Environment Definition --------------------

class MinecraftEnv(gym.Env):
    def __init__(self, action_delay=ACTION_DELAY, max_episode_length=MAX_EPISODE_LENGTH, task_size=TASK_SIZE, reward_model=REWARD_MODEL):
        super(MinecraftEnv, self).__init__()
        self.agent = MinecraftAgent()
        # Include all 25 actions
        self.action_space = spaces.Discrete(25)
        self.log_file = None

        self.task_size = task_size

        # Compute the total size of the flattened observation
        self.position_size = 5  # x, y, z, yaw, pitch
        self.image_size = 224 * 224  # Flattened image size
        self.task_size = task_size  # As defined
        self.scalar_size = 3  # health, hunger, alive
        self.obs_size = self.position_size + self.image_size + self.task_size + self.scalar_size

        # Define the observation space as a Box space with the appropriate shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        # Action mapping - include all 25 actions
        self.action_map = {
            0: 'move_forward',
            1: 'move_backward',
            2: 'strafe_left',
            3: 'strafe_right',
            4: 'turn_left',
            5: 'turn_right',
            6: 'look_up',
            7: 'look_down',
            8: 'jump',
            9: 'toggle_sneak',
            10: 'left_click',
            11: 'right_click',
            12: 'toggle_walk',
            13: 'press_1',
            14: 'press_2',
            15: 'press_3',
            16: 'press_4',
            17: 'press_5',
            18: 'press_6',
            19: 'press_7',
            20: 'press_8',
            21: 'press_9',
            22: 'toggle_space',
            23: 'big_turn_left',
            24: 'big_turn_right'
        }

        self.max_episode_length = max_episode_length
        self.step_counter = 0
        self.action_delay = action_delay
        self.prev_x = 0
        self.prev_z = 0
        self.cumulative_reward = 0
        self.reward_model = reward_model

        self.current_task = np.zeros(task_size, dtype=np.int32)

        # Define reward models
        self.reward_models = {
            "constant_x": self._reward_constant_x,
            "random_task": self._reward_random_task,
            "positive_x": self._reward_positive_x,  # New reward model
            "positive_z": self._reward_positive_z,  # New reward model
            # Add more reward models here
        }

    # -------------------- Reward Functions --------------------

    def _reward_constant_x(self, x, z, delta_x, delta_z, yaw):
        desired_dx = 1  # Always positive X direction
        desired_dz = 0
        desired_yaw = -90

        yaw_diff = abs(yaw - desired_yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)

        progress = (desired_dx * delta_x + desired_dz * delta_z)
        reward = (progress * 3)
        reward += 0.1 * np.cos(np.radians(yaw_diff))

        if progress <= 0:
            reward -= 0.1

        return np.clip(reward, -20.0, 20.0)
    
    def _reward_random_task(self, x, z, delta_x, delta_z, yaw):
        desired_dx, desired_dz = self.current_task[:2]
        if desired_dx != 0 or desired_dz != 0:
            desired_yaw = np.arctan2(desired_dz, desired_dx) * (180 / np.pi)
            if desired_yaw < 0:
                desired_yaw += 360
        else:
            desired_yaw = yaw

        yaw_diff = abs(yaw - desired_yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)

        progress = desired_dx * delta_x + desired_dz * delta_z
        reward = progress
        reward += 0.1 * np.cos(np.radians(yaw_diff))

        if progress <= 0:
            reward -= 0.1

        return np.clip(reward, -10.0, 10.0)
    
    def _reward_positive_x(self, x, z, delta_x, delta_z, yaw):
        # Reward movement in the positive X direction
        reward = delta_x
        return np.clip(reward, -10.0, 10.0)

    def _reward_positive_z(self, x, z, delta_x, delta_z, yaw):
        # Reward movement in the positive Z direction
        reward = delta_z
        return np.clip(reward, -10.0, 10.0)

    def _reward_reach_target(self, x, z, delta_x, delta_z, yaw):
        # Reward the agent for reaching a specific target position
        target_x, target_z = self.current_task[:2]
        distance = np.sqrt((x - target_x)**2 + (z - target_z)**2)
        reward = -distance
        return np.clip(reward, -10.0, 10.0)

    # Add more reward functions as needed

    # -------------------- Gym Methods --------------------

    def preprocess_observation(self, obs):
        # Flatten the observation components into a single array
        position = obs['position'].flatten()
        image = obs['image'].flatten()
        task = obs['task'].flatten()
        health = np.array([obs['health']], dtype=np.float32)
        hunger = np.array([obs['hunger']], dtype=np.float32)
        alive = np.array([obs['alive']], dtype=np.float32)
        flat_obs = np.concatenate([position, image, task, health, hunger, alive])
        # Normalize the observations (optional)
        flat_obs = self.normalize_observation(flat_obs)
        return flat_obs

    def normalize_observation(self, obs):
        # Normalize position values to range [-1, 1]
        obs[:5] /= np.array([20000, 256, 20000, 180, 90])
        # Image values are already between -1 and 1
        # Normalize task values if necessary
        start_idx = self.position_size + self.image_size
        end_idx = start_idx + self.task_size
        obs[start_idx:end_idx] /= 256
        # Normalize health, hunger, alive
        obs[-3] /= 10.0  # health
        obs[-2] /= 10.0  # hunger
        # obs[-1] is alive (0 or 1), so it's already normalized
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())

        action_str = self.action_map.get(action, None)

        # Check if the action is valid
        if action_str is None or not hasattr(self.agent, action_str):
            # Invalid action, apply penalty if desired
            reward = -1.0
            done = False
            truncated = False
            flat_obs = self.preprocess_observation(self.last_obs)
            return flat_obs, reward, done, truncated, {}

        # Execute the action
        getattr(self.agent, action_str)()
        time.sleep(self.action_delay)
        if 'move' in action_str or 'strafe' in action_str:
            self.agent.stop_all_movements()

        # Get the agent's state
        self.agent.get_state()
        x, y, z, yaw, pitch = self.agent.state[:5]

        # Calculate deltas
        delta_x = x - self.prev_x
        delta_z = z - self.prev_z

        # Update previous positions
        self.prev_x = x
        self.prev_z = z

        # Calculate reward using the current reward model
        reward = self.reward_models[self.reward_model](x, z, delta_x, delta_z, yaw)

        
        allowed_actions = {0, 1, 2, 3, 4, 5, 8, 9, 10, 12, 23, 24}
        penalty_actions = {action: -0.2 for action in range(25) if action not in allowed_actions}
        penalty_actions.update({22: -10})

        reward += penalty_actions.get(action, 0.0)
        

        captured_image = self.agent.state[5]

        # Ensure the image has the correct shape before passing it as an observation
        if captured_image is None:
            captured_image = np.zeros((224, 224), dtype=np.float32)
        else:
            captured_image = captured_image.squeeze().cpu().numpy()
            # Ensure image values are between -1 and 1
            captured_image = np.clip(captured_image, -1.0, 1.0)

        # Observation for the model
        obs = {
            "position": np.array([x, y, z, yaw, pitch], dtype=np.float32),
            "image": captured_image,
            "task": self.current_task,
            "health": 10.0,
            "hunger": 10.0,
            "alive": 1,
        }

        flat_obs = self.preprocess_observation(obs)
        self.last_obs = obs  # Save for potential future use

        # Completion checks
        done = self.step_counter >= self.max_episode_length
        truncated = done
        self.cumulative_reward += reward

        # Logging
        self.step_counter += 1

        """
        print(
            f"{Style.BRIGHT}Step {self.step_counter}: "
            f"{Fore.CYAN}X = {x:.2f}{Style.RESET_ALL} | "
            f"{Fore.BLUE}Y = {y:.2f}{Style.RESET_ALL} | "
            f"{Fore.MAGENTA}Z = {z:.2f}{Style.RESET_ALL} | "
            f"{Fore.YELLOW}Yaw = {yaw:.2f}{Style.RESET_ALL} | "
            f"{Fore.GREEN}Pitch = {pitch:.2f}{Style.RESET_ALL} | "
            f"{Fore.RED}Reward = {reward:.4f}{Style.RESET_ALL} | "
            f"{Fore.LIGHTGREEN_EX}Cumulative Reward = {self.cumulative_reward:.2f}{Style.RESET_ALL}",
            end="\r"  # Overwrite the line
        )
        sys.stdout.flush()
        """

        print(f"Step {self.step_counter}: Reward = {reward:.4f}", end='\r')
        sys.stdout.flush()


        log_line = (
            f"Step {self.step_counter}: "
            f"X = {x:.2f}, Y = {y:.2f}, Z = {z:.2f}, "
            f"Yaw = {yaw:.2f}, Pitch = {pitch:.2f}, "
            f"Action = {action}, Action Name = {action_str}, Reward = {reward:.2f}, "
            f"Cumulative Reward = {self.cumulative_reward:.2f}\n"
        )

        self.log_file.write(log_line)
        if done:
            self.log_file.close()

        return flat_obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Get initial state
        self.agent.get_state()
        x, y, z, yaw, pitch = self.agent.state[:5]
        self.prev_x = x
        self.prev_z = z
        self.cumulative_reward = 0

        # Set a new task
        if self.reward_model == "constant_x":
            self.current_task = np.array([1, 0] + [0] * (self.task_size - 2), dtype=np.int32)
        elif self.reward_model == "random_task":
            while True:
                task_values = np.random.randint(-1, 2, size=2, dtype=np.int32)
                if task_values[0] != 0 or task_values[1] != 0:
                    break
            self.current_task = np.concatenate([task_values, np.zeros(self.task_size - 2, dtype=np.int32)])
        elif self.reward_model == "positive_x":
            self.current_task = np.array([1] + [0] * (self.task_size - 1), dtype=np.int32)
        elif self.reward_model == "positive_z":
            self.current_task = np.array([0, 1] + [0] * (self.task_size - 2), dtype=np.int32)
        elif self.reward_model == "reach_target":
            # Set a random target position within a reasonable range
            target_x = x + np.random.uniform(-10, 10)
            target_z = z + np.random.uniform(-10, 10)
            self.current_task = np.array([target_x, target_z] + [0] * (self.task_size - 2), dtype=np.float32)

        print(f"New task values: {self.current_task}")

        # Open a new log file for the episode
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(os.path.join(TRAINING_LOGS_DIR, f"log_{timestamp}.txt"), "w")

        # Get the current screen image or use a blank fallback
        captured_image = self.agent.state[5]
        if captured_image is None:
            captured_image = np.zeros((224, 224), dtype=np.float32)
        else:
            captured_image = captured_image.squeeze().cpu().numpy()
            # Ensure image values are between -1 and 1
            captured_image = np.clip(captured_image, -1.0, 1.0)

        # Reset step counter
        self.step_counter = 0

        # Construct and return the initial observation
        obs = {
            "position": np.array([x, y, z, yaw, pitch], dtype=np.float32),
            "image": captured_image,
            "task": self.current_task,
            "health": 10.0,
            "hunger": 10.0,
            "alive": 1,
        }
        flat_obs = self.preprocess_observation(obs)
        self.last_obs = obs  # Save for potential future use

        # Log the reset event
        print(
            f"{Style.BRIGHT}RESET {self.step_counter}: "
            f"{Fore.CYAN}X = {x:.2f}{Style.RESET_ALL} | "
            f"{Fore.BLUE}Y = {y:.2f}{Style.RESET_ALL} | "
            f"{Fore.MAGENTA}Z = {z:.2f}{Style.RESET_ALL} | "
            f"{Fore.YELLOW}Yaw = {yaw:.2f}{Style.RESET_ALL} | "
            f"{Fore.GREEN}Pitch = {pitch:.2f}{Style.RESET_ALL} | "
            f"{Fore.LIGHTGREEN_EX}Cumulative Reward = {self.cumulative_reward:.2f}{Style.RESET_ALL}"
        )

        return flat_obs, {}
            
    def render(self, mode='human'):
        pass

    def close(self):
        if self.log_file:
            self.log_file.close()


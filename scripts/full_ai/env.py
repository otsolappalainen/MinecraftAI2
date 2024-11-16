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

# -------------------- Configuration --------------------

# Constants
ACTION_DELAY = 0.3
MAX_EPISODE_LENGTH = 250
TASK_SIZE = 20
REWARD_MODEL = "constant_x"  # or "random_task"
TRAINING_LOGS_DIR = "training_logs"
os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)

# -------------------- Environment Definition --------------------

class MinecraftEnv(gym.Env):
    def __init__(self, action_delay=ACTION_DELAY, max_episode_length=MAX_EPISODE_LENGTH, task_size=TASK_SIZE, reward_model=REWARD_MODEL):
        super(MinecraftEnv, self).__init__()
        self.agent = MinecraftAgent()
        self.action_space = spaces.Discrete(25)
        self.log_file = None

        self.task_size = task_size

        # Define the observation space
        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([-20000, -256, -20000, -180, -90]),
                high=np.array([20000, 256, 20000, 180, 90]),
                dtype=np.float32
            ),
            "image": spaces.Box(
                low=-1.0, high=1.0, shape=(224, 224), dtype=np.float32
            ),
            "task": spaces.Box(
                low=-256,
                high=256,
                shape=(self.task_size,),
                dtype=np.int32
            ),
            "health": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "hunger": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "alive": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32)
        })

        # Action mapping 
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

        self.full_action_space = list(range(len(self.action_map)))
        self.current_mask = np.ones(len(self.full_action_space), dtype=np.float32)

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
            "random_task": self._reward_random_task
        }

    # -------------------- Reward Functions --------------------

    def _reward_constant_x(self, x, z, delta_x, delta_z, yaw):
        desired_dx = 1  # Always positive X direction
        desired_dz = 0
        desired_yaw = 0

        yaw_diff = abs(yaw - desired_yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)

        progress = desired_dx * delta_x + desired_dz * delta_z
        reward = progress
        reward += 0.1 * np.cos(np.radians(yaw_diff))

        if progress <= 0:
            reward -= 1.0

        return np.clip(reward, -10.0, 10.0)
    
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

        distance_to_target = np.sqrt(delta_x**2 + delta_z**2)
        reward = -distance_to_target
        reward += 0.1 * np.cos(np.radians(yaw_diff))

        if abs(delta_x) < 0.01 and abs(delta_z) < 0.01:
            reward -= 1.0

        return np.clip(reward, -10.0, 10.0)

    # -------------------- Action Masking --------------------

    def set_action_mask(self, mask):
        assert len(mask) == len(self.full_action_space), "Mask size must match action space size"
        self.current_mask = np.array(mask, dtype=np.float32)

    def get_action_mask(self):
        return self.current_mask

    # -------------------- Gym Methods --------------------

    def step(self, action):
        if not self.current_mask[action]:
            raise ValueError(f"Action {action} is not allowed under the current mask.")

        if isinstance(action, np.ndarray):
            action = int(action.item())

        action_str = self.action_map.get(action, None)
        if action_str and hasattr(self.agent, action_str):
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

        captured_image = self.agent.state[5]

        # Ensure the image has the correct shape before passing it as an observation
        if captured_image is None:
            captured_image = np.zeros((224, 224), dtype=np.float32)
        else:
            captured_image = captured_image.squeeze().cpu().numpy()

        # Observation for the model
        obs = {
            "position": np.array([x, y, z, yaw, pitch], dtype=np.float32),
            "image": captured_image,
            "task": self.current_task,
            "health": 10.0,
            "hunger": 10.0,
            "alive": 1,
            "action_mask": self.current_mask,
        }

        # Completion checks
        done = self.step_counter >= self.max_episode_length
        truncated = done
        self.cumulative_reward += reward

        # Logging
        self.step_counter += 1

        print(
            f"{Style.BRIGHT}Step {self.step_counter}: "
            f"{Fore.CYAN}X = {x:.2f}{Style.RESET_ALL} | "
            f"{Fore.BLUE}Y = {y:.2f}{Style.RESET_ALL} | "
            f"{Fore.MAGENTA}Z = {z:.2f}{Style.RESET_ALL} | "
            f"{Fore.YELLOW}Yaw = {yaw:.2f}{Style.RESET_ALL} | "
            f"{Fore.GREEN}Pitch = {pitch:.2f}{Style.RESET_ALL} | "
            f"{Fore.RED}Reward = {reward:.4f}{Style.RESET_ALL} | "
            f"{Fore.LIGHTGREEN_EX}Cumulative Reward = {self.cumulative_reward:.2f}{Style.RESET_ALL}"
        )

        log_line = (
            f"Step {self.step_counter}: "
            f"X = {x:.2f}, Y = {y:.2f}, Z = {z:.2f}, "
            f"Yaw = {yaw:.2f}, Pitch = {pitch:.2f}, "
            f"Action = {action}, Reward = {reward:.2f}, "
            f"Cumulative Reward = {self.cumulative_reward:.2f}\n"
        )

        self.log_file.write(log_line)
        if done:
            self.log_file.close()

        return obs, reward, done, truncated, {}

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

        # Reset step counter
        self.step_counter = 0

        # Set action mask to allow only specific actions
        allowed_actions = ['move_forward', 'turn_left', 'turn_right']
        self.current_mask = np.zeros(len(self.action_map), dtype=np.float32)
        for action_id, action_name in self.action_map.items():
            if action_name in allowed_actions:
                self.current_mask[action_id] = 1

        print(f"Action mask set to allow: {allowed_actions}")

        # Construct and return the initial observation
        obs = {
            "position": np.array([x, y, z, yaw, pitch], dtype=np.float32),
            "image": captured_image,
            "task": self.current_task,
            "health": 10.0,
            "hunger": 10.0,
            "alive": 1,
            "action_mask": self.current_mask,
        }

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

        return obs, {}
            
    def render(self, mode='human'):
        pass

    def close(self):
        if self.log_file:
            self.log_file.close()


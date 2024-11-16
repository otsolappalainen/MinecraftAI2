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




class MinecraftEnv(gym.Env):
    def __init__(self, action_delay=0.3, max_episode_length=250, task_size=20):
        super(MinecraftEnv, self).__init__()
        self.agent = MinecraftAgent()
        self.action_space = spaces.Discrete(25)  # 12 actions plus toggle walk
        self.log_file = None
        os.makedirs("training_logs", exist_ok=True)

        # Define the observation space with both numeric values and image data
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
                low=-256,  # Allow all values down to -256
                high=256,  # Allow all values up to 256
                shape=(20,),  # Array of length 20
                dtype=np.int32
            ),
            "health": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "hunger": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "alive": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
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

        self.max_episode_length = max_episode_length
        self.step_counter = 0
        self.action_delay = action_delay
        self.prev_x = 0
        self.prev_z = 0
        self.cumulative_reward = 0
        
        # Set a default task array with zero values
        self.current_task = np.zeros(task_size, dtype=np.int32)

    def set_task(self, task_array):
        """Update the task array for different behavior."""
        assert len(task_array) == len(self.current_task), "Task array length mismatch"
        self.current_task = np.array(task_array)
        


    def step(self, action):
        if isinstance(action, np.ndarray):
            print(action)
            action = int(action.item())

        if not hasattr(self, "stuck_action_counter"):
            self.stuck_action_counter = {}
        if not hasattr(self, "stuck_action_threshold"):
            self.stuck_action_threshold = 300  # Adjust this as needed
        
        
        action_str = self.action_map.get(action, None)
        if action_str and hasattr(self.agent, action_str):
            getattr(self.agent, action_str)()
            print(f"action: {action}")
            time.sleep(self.action_delay)
            if 'move' in action_str or 'strafe' in action_str:
                self.agent.stop_all_movements()

        if action not in self.stuck_action_counter:
            self.stuck_action_counter[action] = 0
        self.stuck_action_counter[action] += 1

        if self.stuck_action_counter[action] > self.stuck_action_threshold:
            print(f"Stuck on action {action} for {self.stuck_action_threshold} steps. Resetting environment...")
            obs, _ = self.reset()
            self.stuck_action_counter = {}  # Reset the counter
            return obs, 0.0, True, True, {}


        # Get the agent's state
        self.agent.get_state()
        x, y, z, yaw, pitch = self.agent.state[:5]

        
        # Calculate reward based on task array
        delta_x = x - self.prev_x
        delta_z = z - self.prev_z

        # Update previous positions
        self.prev_x = x
        self.prev_z = z

        # Get task array values
        desired_dx, desired_dz = self.current_task[:2]  # X and Z directions from the task array
        if desired_dx != 0 or desired_dz != 0:  # Ensure there's a valid direction
            desired_yaw = np.arctan2(desired_dz, desired_dx) * (180 / np.pi)
            if desired_yaw < 0:
                desired_yaw += 360
        else:
            desired_yaw = yaw  # Default to current yaw if no direction is set

        yaw_diff = abs(yaw - desired_yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)  # Shortest angular distance

        # Calculate distance moved and raw reward
        distance_to_target = np.sqrt(delta_x**2 + delta_z**2)

        # Reward calculation
        raw_reward = -distance_to_target  # Penalize for distance
        raw_reward += 0.1 * np.cos(np.radians(yaw_diff))  # Reward for alignment with desired yaw

        # Penalize for staying idle
        if abs(delta_x) < 0.01 and abs(delta_z) < 0.01:
            raw_reward -= 1.0

        # Clamp the reward to a safe range
        reward = np.clip(raw_reward, -10.0, 10.0)

        

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
            "health": 10.0,  # Hardcoded health
            "hunger": 10.0,  # Hardcoded hunger
            "alive": 1       # Hardcoded alive
        }
        #print(f"Captured Image Min: {captured_image.min()}, Max: {captured_image.max()}, Type: {captured_image.dtype}")


        """
        if display_image is None:
            display_image = np.zeros((224, 224), dtype=np.uint8)
        else:
            display_image = display_image.squeeze().cpu().numpy()
        # Create a processed copy for visualization
        display_image2 = cv2.cvtColor((display_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        display_image2 = cv2.resize(display_image2, (896, 896), interpolation=cv2.INTER_NEAREST)

        # Add annotations
        cv2.putText(display_image2, f"X: {x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 18, 220), 2)
        cv2.putText(display_image2, f"Y: {y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 18, 220), 2)
        cv2.putText(display_image2, f"Z: {z:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 18, 220), 2)
        cv2.putText(display_image2, f"Yaw: {yaw:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 18, 220), 2)
        cv2.putText(display_image2, f"Pitch: {pitch:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 18, 220), 2)

        # Display the scaled image
        cv2.imshow("Agent State Display", display_image2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        
        """

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
        #print(f"Observation = {obs}")
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

        # Call get_state to obtain the current position at the start of the episode
        self.agent.get_state()
        x, y, z, yaw, pitch = self.agent.state[:5]
        self.prev_x = x
        self.prev_z = z
        self.cumulative_reward = 0

        while True:
            task_values = np.random.randint(-1, 2, size=2, dtype=np.int32)
            if task_values[0] != 0 or task_values[1] != 0:  # Ensure at least one is -1 or 1
                break

        print(f"new task values: {task_values}")
        task_array = np.concatenate([task_values, np.zeros(18, dtype=np.int32)])
        self.current_task = task_array

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"training_logs/log_{timestamp}.txt", "w")

        # Get the current screen image or use a blank fallback
        captured_image = self.agent.state[5]
        
        if captured_image is None:
            captured_image = np.zeros((224, 224), dtype=np.float32)
        else:
            captured_image = captured_image.squeeze().cpu().numpy() 
        

        # Reset step counter and other state-tracking variables
        self.step_counter = 0

        # Construct and return the initial observation
        obs = {
            "position": np.array([x, y, z, yaw, pitch], dtype=np.float32),
            "image": captured_image,
            "task": self.current_task,
            "health": 10.0,
            "hunger": 10.0,
            "alive": 1
        }

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

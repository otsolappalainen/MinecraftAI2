# sim_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import csv

# Constants
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 1  # Grayscale image
TASK_SIZE = 20
ACTION_SPACE_SIZE = 17  # Updated to 17
MAX_EPISODE_LENGTH = 500

INITIAL_HUNGER = 100
INITIAL_HEALTH = 100
INITIAL_ALIVE = 1
YAW_RANGE = (-180, 180)
POSITION_RANGE = (-120, 120)

# Action Definitions
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_MOVE_LEFT = 4
ACTION_MOVE_RIGHT = 5
ACTION_SNEAK_TOGGLE = 6
# Actions 7-16 mapped to "no_op"

# Reward Definitions
REWARD_SCALE_POSITIVE = 1.0  # Reward multiplier for positive movement
REWARD_PENALTY_STAY_STILL = -0.1  # Penalty for non-movement

# Movement Parameters
MOVE_DISTANCE_PER_STEP = 0.25  # Desired position change per timestep
MOVE_DURATION_FORWARD = 10      # Timesteps for moving forward
MOVE_DURATION_SIDES = 4         # Timesteps for moving backward and sideways
ACTION_DURATION = {
    "move_forward": MOVE_DURATION_FORWARD,
    "move_backward": MOVE_DURATION_SIDES,
    "move_left": MOVE_DURATION_SIDES,
    "move_right": MOVE_DURATION_SIDES,
}

class SimulatedEnvSimplified(gym.Env):
    """
    Simulated simplified environment with 2D movement and sneak toggle functionality.
    Observation space is a Dict with 'image' and 'other' to match the full environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        render_mode="none",  # Enforced to "none"
        max_episode_length=MAX_EPISODE_LENGTH,
        log_file="training_data.csv",
        enable_logging=True,
        env_id=0  # Environment ID for logging
    ):
        super(SimulatedEnvSimplified, self).__init__()

        # Enforce render_mode to "none"
        self.render_mode = "none"

        # Define observation space using Dict for image and other data
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32
                ),
                "other": spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32),  # Padded to 28
            }
        )

        # Action space: 17 actions
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Simulation parameters
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.sneaking = False  # Sneak state
        self.hunger = INITIAL_HUNGER
        self.health = INITIAL_HEALTH
        self.alive = INITIAL_ALIVE
        self.cumulative_reward = 0.0
        self.current_task = np.zeros(2, dtype=np.float32)  # Task direction

        self.step_count = 0
        self.max_episode_length = max_episode_length

        # Action Persistence
        self.active_action = None
        self.action_remaining_steps = 0

        # Logging
        self.enable_logging = enable_logging
        if self.enable_logging:
            log_file_name = f"training_data_env_{env_id}.csv"
            self.log_file_handle = open(log_file_name, mode="w", newline="")
            self.log_writer = csv.writer(self.log_file_handle)
            self.log_writer.writerow(["episode_id", "step", "x", "z", "yaw", "reward", "task_x", "task_z"])

        # Initialize RNG
        self.np_random, self.seed_val = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        """
        Sets the seed for this environment's RNG.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.seed_val = seed
        return [seed]

    def _log_step(self, episode_id, step, x, z, yaw, reward, task_x, task_z):
        if self.enable_logging:
            self.log_writer.writerow([episode_id, step, x, z, yaw, reward, task_x, task_z])

    def _get_observation(self):
        image = np.full((IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), 128 / 255.0, dtype=np.float32)  # Constant image
        other = np.zeros(28, dtype=np.float32)
        other[0] = self.x
        other[1] = self.z
        other[2] = self.yaw
        other[3] = self.current_task[0]
        other[4] = self.current_task[1]
        # Remaining 23 elements are zeros
        return {"image": image, "other": other}

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)

        print(f"Resetting environment. Previous cumulative reward: {self.cumulative_reward}")

        # If a new seed is provided, use it
        if seed is not None:
            self.seed(seed)

        self.x = self.np_random.uniform(POSITION_RANGE[0], POSITION_RANGE[1])
        self.z = self.np_random.uniform(POSITION_RANGE[0], POSITION_RANGE[1])
        self.yaw = self.np_random.uniform(YAW_RANGE[0], YAW_RANGE[1])

        self.hunger = INITIAL_HUNGER
        self.health = INITIAL_HEALTH
        self.alive = INITIAL_ALIVE
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.sneaking = False

        # Reset active action
        self.active_action = None
        self.action_remaining_steps = 0

        # Set current task
        possible_tasks = [
            [0, -1],
            [0, 1],
            [1, 0],
            [-1, 0]
        ]
        self.current_task = np.array(random.choice(possible_tasks), dtype=np.float32)

        self._log_step(
            episode_id=0,
            step=self.step_count,
            x=self.x,
            z=self.z,
            yaw=self.yaw,
            reward=0.0,
            task_x=self.current_task[0],
            task_z=self.current_task[1]
        )

        return self._get_observation(), {}

    def _simulate_action(self, action_name):
        """Simulate action execution based on the active action."""
        if self.active_action in ["move_forward", "move_backward", "move_left", "move_right"]:
            speed_factor = 1 / 3.3 if self.sneaking else 1  # SNEAK_FACTOR = 3.3

            # Adjust yaw to wrap around -180 to 180
            self.yaw = (self.yaw + 180) % 360 - 180

            # Convert yaw to radians
            yaw_rad = np.radians(self.yaw)

            # Calculate movement direction
            if self.active_action == "move_forward":
                direction = np.array([np.sin(yaw_rad), -np.cos(yaw_rad)])  # Forward aligns with yaw
            elif self.active_action == "move_backward":
                direction = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])  # Backward opposite to yaw
            elif self.active_action == "move_left":
                direction = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad)])  # Left is 90 degrees counterclockwise
            elif self.active_action == "move_right":
                direction = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])  # Right is 90 degrees clockwise
            else:
                direction = np.array([0.0, 0.0])

            # Apply movement scaling
            delta_position = direction * MOVE_DISTANCE_PER_STEP * speed_factor
            self.x += delta_position[0]
            self.z += delta_position[1]

    def step(self, action):
        self.step_count += 1
        reward = REWARD_PENALTY_STAY_STILL  # Base penalty for every step

        # Action Mapping
        action_map = {
            ACTION_MOVE_FORWARD: "move_forward",
            ACTION_MOVE_BACKWARD: "move_backward",
            ACTION_TURN_LEFT: "turn_left",
            ACTION_TURN_RIGHT: "turn_right",
            ACTION_MOVE_LEFT: "move_left",
            ACTION_MOVE_RIGHT: "move_right",
            ACTION_SNEAK_TOGGLE: "sneak_toggle",
            # Actions 7-16 mapped to "no_op"
        }

        if action not in action_map:
            action_map[action] = "no_op"

        action_name = action_map.get(action, "no_op")

        # Handle action execution
        if action_name in ["move_forward", "move_backward", "move_left", "move_right"]:
            self.active_action = action_name
            self.action_remaining_steps = ACTION_DURATION[action_name]
        elif action_name == "sneak_toggle":
            self.sneaking = not self.sneaking
        elif action_name == "turn_left":
            self.yaw = (self.yaw + 15) % 360  # Increment yaw by 15 degrees
            self.yaw = (self.yaw + 180) % 360 - 180  # Ensure yaw wraps between -180 and 180
        elif action_name == "turn_right":
            self.yaw = (self.yaw - 15) % 360  # Decrement yaw by 15 degrees
            self.yaw = (self.yaw + 180) % 360 - 180  # Ensure yaw wraps between -180 and 180
        elif action_name == "no_op":
            pass

        # Simulate movement based on active action
        if self.active_action is not None and self.action_remaining_steps > 0:
            prev_x, prev_z = self.x, self.z
            self._simulate_action(self.active_action)
            self.action_remaining_steps -= 1
            if self.action_remaining_steps == 0:
                self.active_action = None

            # Calculate movement vector
            delta_x = self.x - prev_x
            delta_z = self.z - prev_z
            movement_vector = np.array([delta_x, delta_z])

            # Normalize task vector
            task_vector = self.current_task / np.linalg.norm(self.current_task)

            # Project movement onto task direction
            movement_along_task = np.dot(movement_vector, task_vector)

            # Reward for correct movement
            reward += movement_along_task

            # Penalize movement in the wrong direction
            movement_off_task = np.linalg.norm(movement_vector) - movement_along_task
            reward -= movement_off_task
        else:
            # No movement action active
            pass

        # Check for episode termination
        terminated = self.step_count >= self.max_episode_length
        truncated = False

        # Get observation
        observation = self._get_observation()

        # Log step every 5 steps
        if self.step_count % 5 == 0:
            self._log_step(
                episode_id=0,
                step=self.step_count,
                x=self.x,
                z=self.z,
                yaw=self.yaw,
                reward=reward,
                task_x=self.current_task[0],
                task_z=self.current_task[1]
            )

        self.cumulative_reward += reward

        return observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        pass  # No rendering in simplified version

    def close(self):
        if self.enable_logging and hasattr(self, 'log_file_handle') and self.log_file_handle:
            self.log_file_handle.close()
        super(SimulatedEnvSimplified, self).close()

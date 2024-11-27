import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import csv
import math

# Constants
TASK_SIZE = 20
ACTION_SPACE_SIZE = 18  # Total number of actions (including unused)
MAX_EPISODE_LENGTH = 500

# Yaw Range in degrees
YAW_RANGE = (-180, 180)

# Action Definitions
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_MOVE_LEFT = 4
ACTION_MOVE_RIGHT = 5
# Actions 6-17 are unused

# Reward Definitions
REWARD_SCALE_POSITIVE = 2.0  # Reward multiplier for positive movement
REWARD_PENALTY_STAY_STILL = -0.1  # Penalty for non-movement

# Movement Parameters
MOVE_DISTANCE_PER_STEP = 0.5  # Desired position change per timestep
TURN_ANGLE_PER_STEP = 15  # Degrees to turn per step

class SimplifiedEnv(gym.Env):
    """
    Simplified environment with 2D omnidirectional movement and randomized initial states.
    Observation space includes position, task array, and status variables.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        render_mode="none",  # Enforced to "none"
        max_episode_length=MAX_EPISODE_LENGTH,
        log_file="training_data.csv",
        enable_logging=True,
        env_id=0,  # Environment ID for logging
    ):
        super(SimplifiedEnv, self).__init__()

        # Enforce render_mode to "none"
        self.render_mode = "none"

        # Define observation space
        # Including status variables and task array
        self.observation_space = spaces.Dict({
            "other": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(27,),  # x, z, yaw, pitch, health, hunger, alive, task (20)
                dtype=np.float32,
            ),
        })

        # Action space: 18 actions (including unused actions)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Simulation parameters
        self.x = 0.0
        self.z = 0.0
        self.yaw = 0.0  # Orientation in degrees
        self.pitch = 0.0  # Kept at 0
        self.health = 20.0
        self.hunger = 20.0
        self.alive = 1.0
        self.cumulative_reward = 0.0
        self.current_task = np.zeros(TASK_SIZE, dtype=np.float32)  # Task direction

        self.step_count = 0
        self.max_episode_length = max_episode_length

        # Logging
        self.env_id = env_id
        self.enable_logging = enable_logging
        if self.enable_logging:
            log_file_name = f"training_data_env_{env_id}.csv"
            self.log_file_handle = open(log_file_name, mode="w", newline="")
            self.log_writer = csv.writer(self.log_file_handle)
            self.log_writer.writerow(
                [
                    "env_id",
                    "episode_id",
                    "step",
                    "x",
                    "z",
                    "yaw",
                    "pitch",
                    "health",
                    "hunger",
                    "alive",
                    "reward",
                    "task_x",
                    "task_z",
                ]
            )

        # Initialize RNG
        self.np_random, self.seed_val = gym.utils.seeding.np_random(None)

    def seed(self, seed=None):
        """
        Sets the seed for this environment's RNG.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.seed_val = seed
        return [seed]

    def _log_step(self, episode_id, step, x, z, yaw, pitch, health, hunger, alive, reward, task_x, task_z):
        if self.enable_logging:
            self.log_writer.writerow(
                [self.env_id, episode_id, step, x, z, yaw, pitch, health, hunger, alive, reward, task_x, task_z]
            )

    def _get_observation(self):
        # Combine all scalar values into 'other'
        other = np.array(
            [
                self.x,
                self.z,
                self.yaw,
                self.pitch,
                self.health,
                self.hunger,
                self.alive
            ] + self.current_task.tolist(),
            dtype=np.float32
        )
        return {"other": other}

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)

        print(
            f"Resetting environment {self.env_id}. Previous cumulative reward: {self.cumulative_reward}"
        )

        # If a new seed is provided, use it
        if seed is not None:
            self.seed(seed)

        # Randomize initial position between -100 and 100
        self.x = self.np_random.uniform(-100, 100)
        self.z = self.np_random.uniform(-100, 100)
        
        # Randomize initial yaw between -180 and 180 degrees
        self.yaw = self.np_random.uniform(YAW_RANGE[0], YAW_RANGE[1])

        self.pitch = 0.0  # Keep pitch at 0
        self.health = 20.0  # Set health to 20
        self.hunger = 20.0  # Set hunger to 20
        self.alive = 1.0  # Set alive to 1

        self.cumulative_reward = 0.0
        self.step_count = 0

        # Set current task
        possible_tasks = [
            [1, 0] + [0]*(TASK_SIZE - 2),   # Positive X direction
            [-1, 0] + [0]*(TASK_SIZE - 2),  # Negative X direction
            [0, 1] + [0]*(TASK_SIZE - 2),   # Positive Z direction
            [0, -1] + [0]*(TASK_SIZE - 2),  # Negative Z direction
        ]
        self.current_task = np.array(random.choice(possible_tasks), dtype=np.float32)

        self._log_step(
            episode_id=0,
            step=self.step_count,
            x=self.x,
            z=self.z,
            yaw=self.yaw,
            pitch=self.pitch,
            health=self.health,
            hunger=self.hunger,
            alive=self.alive,
            reward=0.0,
            task_x=self.current_task[0],
            task_z=self.current_task[1],
        )

        return self._get_observation(), {}

    def _perform_one_step_movement(self, action_name):
        """Perform one-step movement for movement actions based on current yaw."""
        # Convert yaw to radians for trigonometric functions
        yaw_rad = math.radians(self.yaw)

        # Adjust yaw to align forward movement with positive x-axis at yaw = -90
        yaw_rad_adjusted = yaw_rad - math.pi / 2  # Shift yaw by -90 degrees (pi/2 radians)

        # Determine movement direction based on action
        if action_name == "move_forward":
            direction = np.array([math.cos(yaw_rad_adjusted), math.sin(yaw_rad_adjusted)])
        elif action_name == "move_backward":
            direction = np.array([-math.cos(yaw_rad_adjusted), -math.sin(yaw_rad_adjusted)])
        elif action_name == "move_left":
            # Perpendicular to the left of adjusted yaw
            direction = np.array([-math.sin(yaw_rad_adjusted), math.cos(yaw_rad_adjusted)])
        elif action_name == "move_right":
            # Perpendicular to the right of adjusted yaw
            direction = np.array([math.sin(yaw_rad_adjusted), -math.cos(yaw_rad_adjusted)])
        else:
            # No movement for inactive actions
            direction = np.array([0.0, 0.0])

        # Normalize direction vector
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm

        # Apply movement
        delta_position = direction * MOVE_DISTANCE_PER_STEP
        self.x += delta_position[0]
        self.z += delta_position[1]

    def step(self, action):
        self.step_count += 1

        # Initialize immediate_reward
        immediate_reward = REWARD_PENALTY_STAY_STILL  # Base penalty for every step

        # Action Mapping
        action_map = {
            ACTION_MOVE_FORWARD: "move_forward",
            ACTION_MOVE_BACKWARD: "move_backward",
            ACTION_TURN_LEFT: "turn_left",
            ACTION_TURN_RIGHT: "turn_right",
            ACTION_MOVE_LEFT: "move_left",
            ACTION_MOVE_RIGHT: "move_right",
            # Actions 6-17 are unused and mapped to "no_op"
        }

        # Map action index to action name; unused actions do nothing
        action_name = action_map.get(action, "no_op")

        # Handle action execution
        prev_x, prev_z = self.x, self.z
        prev_yaw, prev_pitch = self.yaw, self.pitch

        if action_name in ["move_forward", "move_backward", "move_left", "move_right"]:
            # Perform movement based on current yaw
            self._perform_one_step_movement(action_name)
        elif action_name == "turn_left":
            self.yaw = (self.yaw + TURN_ANGLE_PER_STEP) % 360  # Increment yaw by TURN_ANGLE_PER_STEP degrees
        elif action_name == "turn_right":
            self.yaw = (self.yaw - TURN_ANGLE_PER_STEP) % 360  # Decrement yaw by TURN_ANGLE_PER_STEP degrees
        elif action_name == "no_op":
            pass  # Do nothing for no_op
        else:
            pass  # Future actions can be handled here

        # Ensure yaw is within [-180, 180] degrees for consistency
        if self.yaw > 180:
            self.yaw -= 360
        elif self.yaw <= -180:
            self.yaw += 360

        # Calculate movement vector from position change
        delta_x = self.x - prev_x
        delta_z = self.z - prev_z
        movement_vector = np.array([delta_x, delta_z])

        # Normalize task vector (only use first 2 elements)
        task_vector = self.current_task[:2]
        task_norm = np.linalg.norm(task_vector)
        if task_norm == 0:
            task_vector_normalized = np.array([0.0, 0.0])
        else:
            task_vector_normalized = task_vector / task_norm

        # Project movement onto task direction
        movement_along_task = np.dot(movement_vector, task_vector_normalized)

        # Calculate reward based on movement alignment
        movement_reward = movement_along_task * REWARD_SCALE_POSITIVE
        total_reward = immediate_reward + movement_reward

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
                pitch=self.pitch,
                health=self.health,
                hunger=self.hunger,
                alive=self.alive,
                reward=total_reward,
                task_x=self.current_task[0],
                task_z=self.current_task[1],
            )

        self.cumulative_reward += total_reward

        return observation, total_reward, terminated, truncated, {}

    def render(self, mode="human"):
        pass  # No rendering in simplified version

    def close(self):
        if self.enable_logging and hasattr(self, "log_file_handle") and self.log_file_handle:
            self.log_file_handle.close()
        super(SimplifiedEnv, self).close()


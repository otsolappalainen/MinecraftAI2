import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time





class SimulatedEnvGraphics(gym.Env):
    """
    Simulated environment with lightweight graphics for visualization and
    compatibility with the real environment.
    """
    def __init__(self, render_mode="none", grid_size=500, cell_size=50, task_size=20, max_episode_length=1000, simulation_speed=60, zoom_factor=0.5):
        super(SimulatedEnvGraphics, self).__init__()

        # Match observation space with the real environment
        self.position_size = 5  # x, y, z, yaw, pitch
        self.image_size = 224 * 224  # Flattened image size
        self.scalar_size = 3  # health, hunger, alive
        self.task_size = task_size
        self.obs_size = self.position_size + self.image_size + self.scalar_size + self.task_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        # Match action space with the real environment
        self.action_space = spaces.Discrete(25)  # Include all 25 actions

        # Simulation parameters
        self.x = 0
        self.z = 0
        self.yaw = 0
        self.pitch = 0
        self.hunger = 100
        self.health = 100
        self.alive = 1
        self.cumulative_reward = 0
        self.current_task = np.zeros(task_size, dtype=np.float32)
        self.current_task[0] = 1  # Set the first task to 1
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.zoom_factor = zoom_factor

        self.step_count = 0
        self.max_episode_length = max_episode_length
        self.render_mode = render_mode
        self.simulation_speed = simulation_speed  # Updates per second
        self.time_step = 1.0 / self.simulation_speed  # Time interval per update
        self.window_size = min(grid_size, 1000)  # Limit rendering window size to 1000x1000
        self.window = None
        self.clock = None

    def _get_observation(self):
        """
        Create an observation that matches the real environment's format.
        """
        # Dynamically generate a blank grayscale image (224x224)
        image_flattened = np.full((224 * 224,), 128, dtype=np.uint8) / 255.0  # Gray image normalized to 0-1

        # Positional and scalar values
        positional_values = np.array([self.x, 0, self.z, self.yaw, self.pitch], dtype=np.float32)
        scalar_values = np.array([self.hunger, self.health, self.alive], dtype=np.float32)

        # Combine into a single observation
        observation = np.concatenate([image_flattened, positional_values, scalar_values, self.current_task])
        return observation

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)

        print(f"reward {self.cumulative_reward}")


        # Reset simulation parameters
        self.x = np.random.uniform(-400, 400)
        self.z = np.random.uniform(-400, 400)

        self.yaw = np.random.uniform(0, 360)  # Random yaw (0 to 360 degrees)
        self.pitch = np.random.uniform(-90, 90)  # Random pitch (-90 to 90 degrees)


        self.hunger = 100
        self.health = 100
        self.alive = 1
        self.step_count = 0
        self.prev_x = self.x  # Initialize prev_x to the starting x position
        self.cumulative_reward = 0

        if self.render_mode == "human":
            self._initialize_graphics()

        return self._get_observation(), {}

    def step(self, action):
        """
        Perform a step in the environment.
        """
        self.step_count += 1

        # Actions (only a subset affects the simulation; others are no-ops)
        if action == 0:  # Move forward
            self.x += np.sin(np.radians(self.yaw))
            self.z += np.cos(np.radians(self.yaw))
        elif action == 1:  # Move backward
            self.x -= np.sin(np.radians(self.yaw))
            self.z -= np.cos(np.radians(self.yaw))
        elif action == 2:  # Turn left
            self.yaw = (self.yaw - 10) % 360
        elif action == 3:  # Turn right
            self.yaw = (self.yaw + 10) % 360
        # Other actions do nothing (no-op)

        # Calculate reward for moving towards increasing x
        delta_x = self.x - self.prev_x
        reward = delta_x * 2
        self.cumulative_reward += reward

        # Update previous x
        self.prev_x = self.x

        # Determine if the episode is done
        terminated = self.step_count >= self.max_episode_length
        truncated = False

        # Render if in graphical mode
        if self.render_mode == "human":
            self._render()
            time.sleep(self.time_step)

        # Return the updated observation, reward, and status flags
        return self._get_observation(), reward, terminated, truncated, {}

    def _initialize_graphics(self):
        """
        Initialize the graphical environment.
        """
        pygame.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Simulated Environment")
        self.clock = pygame.time.Clock()

    def _render(self):
        """
        Render the environment using pygame with a zoom factor.
        """
        if self.window is None:
            self._initialize_graphics()

        # Clear screen (white)
        self.window.fill((255, 255, 255))

        # Draw grid based on zoom factor
        adjusted_cell_size = self.cell_size * self.zoom_factor
        for x in range(0, self.window_size, int(adjusted_cell_size)):
            pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, int(adjusted_cell_size)):
            pygame.draw.line(self.window, (200, 200, 200), (0, y), (self.window_size, y))

        # Convert x, z to screen coordinates relative to the center of the grid
        center_offset = self.grid_size // 2
        agent_x = int(
            (self.x + center_offset) * self.zoom_factor * self.window_size / self.grid_size
        )
        agent_z = int(
            (self.z + center_offset) * self.zoom_factor * self.window_size / self.grid_size
        )

        # Draw agent (triangle arrow)
        agent_direction = np.radians(self.yaw)
        agent_triangle = [
            (agent_x + 10 * self.zoom_factor * np.cos(agent_direction), agent_z + 10 * self.zoom_factor * np.sin(agent_direction)),
            (agent_x + 5 * self.zoom_factor * np.cos(agent_direction + 2.5), agent_z + 5 * self.zoom_factor * np.sin(agent_direction + 2.5)),
            (agent_x + 5 * self.zoom_factor * np.cos(agent_direction - 2.5), agent_z + 5 * self.zoom_factor * np.sin(agent_direction - 2.5)),
        ]
        pygame.draw.polygon(self.window, (0, 0, 255), agent_triangle)

        # Render text for x, z coordinates and cumulative reward
        font = pygame.font.SysFont(None, 24)
        text = font.render(
            f"X: {self.x:.2f}, Z: {self.z:.2f}, Reward: {self.cumulative_reward:.2f}",
            True,
            (0, 0, 0),
        )
        text_rect = text.get_rect(center=(agent_x, agent_z - 15 * self.zoom_factor))
        self.window.blit(text, text_rect)

        # Refresh screen
        pygame.display.flip()
        self.clock.tick(self.simulation_speed)


    
    def close(self):
        """
        Close the graphical environment.
        """
        if self.window:
            pygame.quit()
            self.window = None

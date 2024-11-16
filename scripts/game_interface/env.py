import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from agent import SimulatedAgent
import pygame
from gym.utils import seeding
from datetime import datetime
import os
import csv

class SimpleMinecraftEnv(gym.Env):
    def __init__(self, render_delay=0.1, speed_factor=1.0, plot_update_freq=100, 
                 enable_rendering=False, enable_plot=False, max_episode_length=1000, 
                 arena_size=20, noise_level=0.0, target_distance=10,
                 random_target_mode=True, target_coordinates=(0, 0), target_area_size=5):
        super(SimpleMinecraftEnv, self).__init__()

        # Set max_episode_length from the parameter
        self.max_episode_length = max_episode_length

        # Initialize the action and observation spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array([-100000, -100000, -180, -90, -100000, -100000]),  # add goal x, z
            high=np.array([100000, 100000, 180, 90, 100000, 100000]),
            dtype=np.float32
        )

        # Set difficulty-specific parameters
        self.arena_size = arena_size
        self.noise_level = noise_level
        self.target_distance = target_distance
        self.random_target_mode = random_target_mode
        self.target_coordinates = target_coordinates
          # Default to (0, 0) or as specified
        self.target_area_size = target_area_size

        self.trial_number = 0  # Initialize trial number
        self.episode_number = 0  # Initialize episode number

        # Generate unique file names for logs based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = f"training_log_{timestamp}.txt"
        self.episode_rewards_log = f"episode_rewards_log_{timestamp}.csv"

        # Initialize the episode rewards log with headers
        with open(self.episode_rewards_log, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Trial", "Episode", "Cumulative Reward"])

        # Write the initial log file for other logs
        with open(self.log_file_path, "w") as f:
            f.write("Episode Log\n")

        # Rendering and agent attributes
        self.render_delay = render_delay
        self.speed_factor = speed_factor
        self.enable_rendering = enable_rendering
        self.enable_plot = enable_plot
        
        self.step_counter = 0
        self.total_steps = 0
        self.action_history = []
        self.reward_history = []
        self.seed_value = None
        self.halfway_reward_given = False
        self.quarterway_reward_given = False

        # Initialize Pygame if rendering is enabled
        if self.enable_rendering:
            pygame.init()
            self.screen_size = (600, 600)
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        # Initialize agent and target position
        self.agent = SimulatedAgent()  # Ensure agent is properly initialized here
        self.target_x, self.target_z = 0, 0  # Default target location; set in reset
        self.prev_distance_to_target = None

        # Initialize plot if explicitly enabled
        if self.enable_plot:
            self.init_plot()

    def log_episode_reward(self, cumulative_reward):
        """Log the cumulative reward for an episode to a CSV file."""
        with open(self.episode_rewards_log, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.trial_number, self.episode_number, cumulative_reward])

    def log(self, message):
        """Log message to file and optionally to console."""
        with open(self.log_file_path, "a") as f:
            f.write(message + "\n")

    def init_plot(self):
        """Initialize the plot for rendering."""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-self.observation_space.high[0], self.observation_space.high[0])
        self.ax.set_ylim(-self.observation_space.high[1], self.observation_space.high[1])
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Z Position")
        self.ax.set_title("Agent Position and Target")

    def step(self, action):
        """Apply action to the agent and return new observation."""
        if hasattr(self, 'reached_target') and self.reached_target:
            # If agent has reached target, remain in place with zero reward
            x, z, yaw = self.agent.get_position()
            pitch = 0  # Placeholder for pitch
            obs = np.array([x, z, yaw, pitch, self.target_x, self.target_z], dtype=np.float32)
            truncated = False
            return obs, 0, True, truncated, {}

        # Perform action with optional noise
        if action == 0:
            self.agent.move_forward(noise_level=self.noise_level)
        elif action == 1:
            self.agent.move_backward(noise_level=self.noise_level)
        elif action == 2:
            self.agent.strafe_left(noise_level=self.noise_level)
        elif action == 3:
            self.agent.strafe_right(noise_level=self.noise_level)
        elif action == 4:
            self.agent.turn_left()
        elif action == 5:
            self.agent.turn_right()

        # Get agent's current position and yaw
        x, z, yaw = self.agent.get_position()
        pitch = 0  # Placeholder for pitch

        # Calculate distance to the target
        new_distance_to_target = np.sqrt((self.agent.x - self.target_x) ** 2 + (self.agent.z - self.target_z) ** 2)
        old_distance_to_target = self.prev_distance_to_target
        self.prev_distance_to_target = new_distance_to_target

        # Define target area size (proportional to arena size)
        
        done = new_distance_to_target < self.target_area_size  # Check if within target area
        reward = 0
        truncated = False  # Initialize truncated here

        if done:
            reward = 1000  # Large reward for reaching the target
            self.reached_target = True
        elif self.step_counter >= self.max_episode_length:
            done = True
            reward = -100  # Penalty for exceeding max steps
            truncated = True
        else:
            # Reward for moving closer to the target
            reward += (old_distance_to_target - new_distance_to_target) - 0.3

            # Penalty for moving out of bounds
            if np.abs(self.agent.x) > self.arena_size / 2 or np.abs(self.agent.z) > self.arena_size / 2:
                reward -= 10

            # Bonus rewards for reaching milestones without exploitation
            if not self.halfway_reward_given and new_distance_to_target <= old_distance_to_target * 0.5:
                reward += 50  # Bonus for reaching halfway
                self.halfway_reward_given = True

            if not self.quarterway_reward_given and new_distance_to_target <= old_distance_to_target * 0.25:
                reward += 100  # Bonus for reaching quarterway
                self.quarterway_reward_given = True

        obs = np.array([x, z, yaw, pitch, self.target_x, self.target_z], dtype=np.float32)
        self.reward_history.append(reward)
        self.step_counter += 1
        self.total_steps += 1

        return obs, reward, done, truncated, {}


    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if self.reward_history:
            cumulative_reward = sum(self.reward_history)
            with open(self.episode_rewards_log, "a") as f:
                f.write(f"{self.trial_number},{self.episode_number},{cumulative_reward}\n")
            self.episode_number += 1
        
        self.halfway_reward_given = False
        self.quarterway_reward_given = False
        self.prev_distance_to_target = np.inf 

        # Set target coordinates based on random_target_mode
        if self.random_target_mode:
            self.target_x = np.random.uniform(-self.arena_size / 2, self.arena_size / 2)
            self.target_z = np.random.uniform(-self.arena_size / 2, self.arena_size / 2)
        else:
            self.target_x, self.target_z = self.target_coordinates

        # Define a random radius and angle to position the agent around the target
        target_radius = np.random.uniform(0, self.arena_size / 2)
        angle = np.random.uniform(0, 2 * np.pi)
        spawn_x = self.target_x + target_radius * np.cos(angle)
        spawn_z = self.target_z + target_radius * np.sin(angle)
        yaw = np.random.uniform(-180, 180)

        self.agent.reset(x=spawn_x, z=spawn_z, yaw=yaw)
        self.reward_history = []
        self.step_counter = 0

        self.prev_distance_to_target = np.sqrt(
            (self.agent.x - self.target_x) ** 2 + (self.agent.z - self.target_z) ** 2
        )
        self.reached_target = False

        return np.array([self.agent.x, self.agent.z, self.agent.yaw, 0, self.target_x, self.target_z], dtype=np.float32), {}



    def render(self, mode='human'):
        if not self.enable_rendering:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

        self.screen.fill((0, 0, 0))
        x, z, yaw = self.agent.get_position()
        pygame.draw.circle(self.screen, (255, 0, 0), (int(x + self.screen_size[0] / 2), int(z + self.screen_size[1] / 2)), 10)
        end_pos = (int(x + 20 * np.cos(np.radians(yaw)) + self.screen_size[0] / 2), int(z + self.screen_size[1] / 2))
        pygame.draw.line(self.screen, (0, 255, 0), (int(x + self.screen_size[0] / 2), int(z + self.screen_size[1] / 2)), end_pos, 5)
        pygame.draw.circle(self.screen, (0, 0, 255), (int(self.target_x + self.screen_size[0] / 2), int(self.target_z + self.screen_size[1] / 2)), 10)

        pygame.display.flip()
        self.clock.tick(60)



    def seed(self, seed=None):
        self.seed_value = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close(self):
            if self.enable_rendering:
                pygame.quit()
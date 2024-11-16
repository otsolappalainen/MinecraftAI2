import numpy as np
import os
import glob
import time
import pygame
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import SimpleMinecraftEnv


difficulty_settings = {
    "easy": {
        "arena_size": 20,
        "noise_level": 0.0,
        "target_distance": 10,
        "target_area_size": 5,
        "random_target_mode": False,
        "target_coordinates": (0, 0)
    },
    "medium": {
        "arena_size": 50,
        "noise_level": 0.05,
        "target_distance": 25,
        "target_area_size": 10,
        "random_target_mode": False,
        "target_coordinates": (10, 10)
    },
    "hard": {
        "arena_size": 100,
        "noise_level": 0.1,
        "target_distance": 50,
        "target_area_size": 15,
        "random_target_mode": True
    }
}

def record_trajectories(
    model_path, 
    log_dir="./trajectories", 
    num_agents=1, 
    difficulty="easy"
):
    os.makedirs(log_dir, exist_ok=True)
    
    # Retrieve settings for the specified difficulty level
    settings = difficulty_settings.get(difficulty, difficulty_settings["easy"])
    
    # Wrap the environment in DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([
        lambda: SimpleMinecraftEnv(
            enable_rendering=False,
            enable_plot=False,
            arena_size=settings["arena_size"],
            noise_level=settings["noise_level"],
            target_distance=settings["target_distance"],
            target_area_size=settings["target_area_size"],
            random_target_mode=settings["random_target_mode"],
            target_coordinates=settings["target_coordinates"]
        )
    ])

    # Load the pre-trained model
    if os.path.isfile(model_path):
        print(f"Loading the existing model from {model_path}...")
        model = PPO.load(model_path, env=env, verbose=1)
    else:
        print("No existing model found at the specified path.")
        return

    trajectories = []

    # Loop through the specified number of episodes for recording
    for episode in range(num_agents):
        obs = env.reset()
        done = False
        episode_trajectory = []

        # Retrieve target position from the initial observation
        try:
            target_x, target_z = obs[0][4], obs[0][5]
        except IndexError as e:
            print(f"Error accessing target position in observation: {obs}")
            env.close()
            return

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            try:
                x, z, yaw, pitch = obs[0][:4]
            except IndexError as e:
                print(f"Error unpacking observation values: {obs}")
                env.close()
                return

            # Record raw coordinates without scaling
            episode_trajectory.append((x, z, yaw, pitch, target_x, target_z))
        
        trajectories.append(episode_trajectory)

    timestamp = int(time.time())
    save_path = os.path.join(log_dir, f"trajectories_{timestamp}.npy")
    np.save(save_path, np.array(trajectories, dtype=object))
    print(f"Trajectories saved to {save_path}.")
    env.close()



def replay_trajectories(
    trajectory_path, 
    screen_size=(600, 600)
):
    # Load trajectories
    trajectories = np.load(trajectory_path, allow_pickle=True).tolist()
    num_agents = len(trajectories)

    # Initialize pygame for rendering
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    
    # Initialize font for text rendering
    font = pygame.font.Font(None, 16)

    # Determine the arena size from the first trajectory and calculate the scale factor
    if trajectories:
        max_coord = max(max(abs(x), abs(z)) for trajectory in trajectories for (x, z, *_rest) in trajectory)
        scale_factor = screen_size[0] / (2 * max_coord)
    
    # Initialize colors for agents and target reach tracking
    colors = [(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ) for _ in range(num_agents)]
    
    # Initialize agents' state
    agents = []
    for i in range(num_agents):
        trajectory = trajectories[i]
        target_x, target_z = trajectory[0][4], trajectory[0][5]
        prev_distance_to_target = np.sqrt((trajectory[0][0] - target_x) ** 2 + (trajectory[0][1] - target_z) ** 2)
        
        agents.append({
            "trajectory": trajectory,
            "color": colors[i],
            "step": 0,
            "target_position": (target_x, target_z),
            "cumulative_reward": 0,
            "prev_distance_to_target": prev_distance_to_target,
            "reached_target": False
        })

    timestep = 0  # Initialize timestep counter
    arena_half_size = int(scale_factor * max_coord)
    agents_reached_target = 0

    # Replay loop
    running = True
    while running:
        screen.fill((0, 0, 0))

        # Draw the arena boundary
        pygame.draw.rect(
            screen, 
            (255, 255, 255), 
            pygame.Rect(
                (screen_size[0] // 2 - arena_half_size, screen_size[1] // 2 - arena_half_size), 
                (arena_half_size * 2, arena_half_size * 2)
            ), 
            1
        )

        # Update agents' positions and draw them
        for agent in agents:
            trajectory = agent["trajectory"]
            step = agent["step"]

            if step < len(trajectory):
                x, z = trajectory[step][:2]
                target_x, target_z = agent["target_position"]

                # Scale agent and target positions to fit within screen space
                screen_x = int((x * scale_factor) + screen_size[0] / 2)
                screen_z = int((z * scale_factor) + screen_size[1] / 2)
                screen_target_x = int((target_x * scale_factor) + screen_size[0] / 2)
                screen_target_z = int((target_z * scale_factor) + screen_size[1] / 2)

                # Draw the target position as a blue circle
                pygame.draw.circle(screen, (0, 0, 255), (screen_target_x, screen_target_z), 5)

                # Calculate reward based on distance to target
                new_distance_to_target = np.sqrt((x - target_x) ** 2 + (z - target_z) ** 2)
                reward = 0
                done = False

                if new_distance_to_target < 5:  # Assuming a target area size of 5
                    reward = 1000
                    agent["reached_target"] = True
                    done = True
                    agents_reached_target += 1
                elif step == len(trajectory) - 1:
                    reward = -100  # Penalty for not reaching the target
                else:
                    reward = (agent["prev_distance_to_target"] - new_distance_to_target) - 0.3
                    if abs(x) > max_coord or abs(z) > max_coord:
                        reward -= 10  # Penalty for moving out of bounds

                # Update cumulative reward and previous distance
                agent["cumulative_reward"] += reward
                agent["prev_distance_to_target"] = new_distance_to_target

                # Draw the agent's position as a small circle with cumulative reward
                pygame.draw.circle(screen, agent["color"], (screen_x, screen_z), 3)
                reward_text = font.render(f"{int(agent['cumulative_reward'])}", True, (255, 255, 255))
                screen.blit(reward_text, (screen_x + 10, screen_z - 10))

                # Increment step if agent hasn't reached the target
                if not agent["reached_target"]:
                    agent["step"] += 1

        # Render the timestep counter and agents reached target
        timestep_text = font.render(f"Timestep: {timestep}", True, (255, 255, 255))
        screen.blit(timestep_text, (10, 10))
        
        reached_text = font.render(f"Reached: {agents_reached_target}/{num_agents}", True, (0, 255, 0))
        screen.blit(reached_text, (10, 50))
        
        timestep += 1

        # Check for window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(30)  # Adjust playback speed

    pygame.quit()



# Example main function to run recording or replay
def main():
    choice = input("Do you want to (1) Record a new trajectory or (2) Play back an existing recording? Enter 1 or 2: ").strip()
    if choice == "1":
        base_directory = "C:\\Users\\odezz\\source\\MinecraftAI2\\scripts\\game_interface\\logs"
        difficulty_levels = ["easy", "medium", "hard"]
        model_files = []

        # Collect models from each difficulty folder
        for difficulty in difficulty_levels:
            model_files.extend(glob.glob(os.path.join(base_directory, difficulty, "*.zip")))

        if not model_files:
            print("No models found in any difficulty folder.")
            return
        
        print("Available models:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}: {model_file}")
        model_choice = int(input("Select a model to use for recording (enter number): ")) - 1
        model_path = model_files[model_choice]

        num_agents = int(input("Enter the number of agents to record trajectories for: "))
        difficulty = input("Enter difficulty level (easy, medium, hard): ").strip().lower()

        record_trajectories(
            model_path=model_path, 
            log_dir="./trajectories", 
            num_agents=num_agents, 
            difficulty=difficulty
        )
    elif choice == "2":
        trajectory_files = glob.glob("./trajectories/trajectories_*.npy")
        if not trajectory_files:
            print("No trajectory recordings found.")
            return

        print("Available recordings:")
        for i, trajectory_file in enumerate(trajectory_files, 1):
            print(f"{i}: {trajectory_file}")
        recording_choice = int(input("Select a recording to play back (enter number): ")) - 1
        trajectory_path = trajectory_files[recording_choice]

        replay_trajectories(trajectory_path=trajectory_path)
    else:
        print("Invalid choice. Exiting.")

if __name__ == '__main__':
    main()
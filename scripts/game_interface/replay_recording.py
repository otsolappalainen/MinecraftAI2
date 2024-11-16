import numpy as np
import pygame
import random

def replay_trajectories(
    trajectory_path="./trajectories/trajectories.npy", 
    num_agents=1, 
    screen_size=(1200, 1200),  # Larger window size for better visibility
    scale_factor=1  # Adjust scale factor as needed
):
    # Load trajectories
    trajectories = np.load(trajectory_path, allow_pickle=True).tolist()  # Convert to a list
    
    # Initialize pygame for rendering
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()
    
    # Initialize font for timestep counter
    font = pygame.font.Font(None, 36)  # Use a default font with size 36

    # Initialize agents, each with a unique color and assigned trajectory
    agents = []
    colors = [(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ) for _ in range(num_agents)]
    
    for i in range(num_agents):
        trajectory = random.choice(trajectories)
        # Check if the target position is available in the trajectory
        if len(trajectory[0]) >= 6:
            target_position = (trajectory[0][4], trajectory[0][5])  # Extract target position from the first step
        else:
            target_position = None  # No target recorded for this trajectory
        
        agents.append({
            "trajectory": trajectory,
            "color": colors[i],
            "step": 0,
            "target_position": target_position
        })

    timestep = 0  # Initialize timestep counter

    # Replay loop
    running = True
    while running:
        screen.fill((0, 0, 0))

        # Update agents' positions and draw them
        for agent in agents:
            trajectory = agent["trajectory"]
            step = agent["step"]

            if step < len(trajectory):
                # Unpack agent's position (x, z) and target position (target_x, target_z if available)
                x, z = trajectory[step][:2]
                
                # Draw the target position if available, with the agent's color
                if agent["target_position"] is not None:
                    target_x, target_z = agent["target_position"]
                    # Adjust target position to screen space
                    screen_target_x = int((target_x * scale_factor) + screen_size[0] // 2)
                    screen_target_z = int((target_z * scale_factor) + screen_size[1] // 2)
                    
                    pygame.draw.circle(screen, agent["color"], (screen_target_x, screen_target_z), 8)
                    #print(f"Target (x, z): ({target_x}, {target_z}) -> Screen (x, z): ({screen_target_x}, {screen_target_z})")

                # Draw the agent's position after adjusting for screen center and scaling
                screen_x = int((x * scale_factor) + screen_size[0] // 2)
                screen_z = int((z * scale_factor) + screen_size[1] // 2)
                
                pygame.draw.circle(screen, agent["color"], (screen_x, screen_z), 3)
                #print(f"Agent (x, z): ({x}, {z}) -> Screen (x, z): ({screen_x}, {screen_z})")
                
                agent["step"] += 1  # Move to the next step in the trajectory
            #else:
                #print(f"Agent {agents.index(agent)} has finished its trajectory.")

        # Render the timestep counter in the top right corner
        timestep_text = font.render(f"Timestep: {timestep}", True, (255, 255, 255))
        screen.blit(timestep_text, (screen_size[0] - timestep_text.get_width() - 10, 10))

        # Increment timestep counter
        timestep += 1

        # Check for window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.flip()
        clock.tick(30)  # Adjust playback speed

    pygame.quit()

if __name__ == '__main__':
    replay_trajectories()

# test_env.py

import gymnasium as gym
from sim_env import SimulatedEnvSimplified  # Ensure this import matches your project structure
import time
import logging
import random

def setup_logging():
    """
    Configures the logging settings.
    Logs are displayed on the console with timestamps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()  # Log to console
            # To also log to a file, uncomment the following line:
            # logging.FileHandler("env_test.log"),
        ]
    )

def main():
    """
    Main function to run the testing loops.
    Executes Loop1 and Loop2 sequentially and then terminates.
    """
    setup_logging()
    logging.info("Initializing the SimulatedEnvSimplified environment...")
    
    # Initialize the environment
    env = SimulatedEnvSimplified(env_id=0)  # Assign a unique ID if running multiple instances
    env.seed(42)  # Seed for reproducibility
    observation, info = env.reset()
    logging.info("Environment reset. Starting tests.")
    
    # Define action mapping
    action_map = {
        "move_forward": 0,
        "move_backward": 1,
        "turn_left": 2,
        "turn_right": 3,
        "move_left": 4,
        "move_right": 5,
        "sneak_toggle": 6,
        "no_op": 7,  # Actions 7-16 are "no_op"
    }
    
    # Define a list of non-movement actions
    non_movement_actions = ["turn_left", "turn_right", "sneak_toggle", "no_op"]
    
    try:
        # ===== Loop1 =====
        logging.info("===== Starting Loop1 =====")
        # Send "move_forward" action
        action_name = "move_forward"
        action_id = action_map.get(action_name, 7)  # Default to "no_op" if not found
        send_action(env, action_id, action_name)
        
        # Send 15 non-movement actions
        for i in range(15):
            action_name = random.choice(non_movement_actions)
            action_id = action_map.get(action_name, 7)  # Default to "no_op"
            send_action(env, action_id, action_name)
        
        # ===== Loop2 =====
        logging.info("===== Starting Loop2 =====")
        # Send "move_forward" action
        action_name = "move_forward"
        action_id = action_map.get(action_name, 7)
        send_action(env, action_id, action_name)
        
        # Send 2 non-movement actions
        for i in range(2):
            action_name = random.choice(non_movement_actions)
            action_id = action_map.get(action_name, 7)
            send_action(env, action_id, action_name)
        
        # Send "move_backward" action
        action_name = "move_backward"
        action_id = action_map.get(action_name, 7)
        send_action(env, action_id, action_name)
        
        # Send 5 more non-movement actions
        for i in range(5):
            action_name = random.choice(non_movement_actions)
            action_id = action_map.get(action_name, 7)
            send_action(env, action_id, action_name)
        
        logging.info("===== Testing Completed =====")
    
    except Exception as e:
        logging.error(f"An error occurred during testing: {e}")
    
    finally:
        env.close()
        logging.info("Environment closed.")

def send_action(env, action_id, action_name):
    """
    Sends an action to the environment, logs the time before and after,
    and logs the response details.
    
    Parameters:
    - env: The Gym environment.
    - action_id: Integer ID of the action.
    - action_name: String name of the action.
    """
    # Log time before sending the action
    time_before = time.time()
    logging.info(f"Sending action: {action_name} (ID: {action_id}) at {time_before:.6f}")
    
    # Send action to the environment
    observation, reward, done, truncated, info = env.step(action_id)
    
    # Log time after receiving the response
    time_after = time.time()
    logging.info(f"Received response at {time_after:.6f}. Reward: {reward}, Position: ({env.x:.3f}, {env.z:.3f}), Yaw: {env.yaw:.1f}")
    
    # Optionally, log additional info if available
    if "action" in info:
        logging.info(f"Info: Action performed: {info['action']}")
    if "reward" in info:
        logging.info(f"Info: Reward received: {info['reward']}")
    if "x_position" in info and "z_position" in info:
        logging.info(f"Info: X Position: {info['x_position']:.3f}, Z Position: {info['z_position']:.3f}")
    
    # Check if the episode has ended
    if done or truncated:
        logging.info("Episode terminated. Resetting environment...")
        observation, info = env.reset()
        logging.info("Environment reset after termination.")
    
    # Introduce a 1-second delay
    time.sleep(1)

if __name__ == "__main__":
    main()

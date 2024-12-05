import os
import time
import random
import json
from datetime import datetime
import numpy as np
import logging
from env_dqn_paraller import MinecraftEnv

def serialize_observation(observation):
    """Convert numpy arrays in observation to lists for JSON serialization."""
    serialized = {}
    for k, v in observation.items():
        if isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        elif isinstance(v, dict):
            serialized[k] = {sk: sv.tolist() if isinstance(sv, np.ndarray) else sv 
                           for sk, sv in v.items()}
        else:
            serialized[k] = v
    return serialized

def main():
    # Set up logging
    # Clear existing log file
    with open('mock_train.log', 'w') as f:
        f.write('')  # Clear file
        
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mock_train.log'),
            logging.StreamHandler()
        ]
    )

    # Configuration
    uris = [
        "ws://localhost:8080",
        "ws://localhost:8081",
        "ws://localhost:8082",
        "ws://localhost:8083"
    ]
    num_steps = 40
    actions_per_second = 1
    log_dir = "env_logs"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create log directory with timestamp
    log_dir = os.path.join(log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize log_files list outside try block
    log_files = []

    try:
        # Initialize environment with desired number of URIs
        logging.info("Initializing environment...")
        env = MinecraftEnv(uris=uris[:3])  # Using 2 clients
        
        # Wait for connections to establish
        time.sleep(1)  # Add delay to ensure connections are ready
        
        # Open log files for each client
        for i in range(env.num_clients):
            log_path = os.path.join(log_dir, f"client_{i}_log.jsonl")
            f = open(log_path, 'w', encoding='utf-8')
            headers = [
                "Step", "Timestamp_Action_Sent", "Action", "Action_Name",
                "Timestamp_Received", "Duration_sec", "Reward", "Done",
                "Info", "Observation"
            ]
            f.write(",".join(headers) + "\n")
            log_files.append(f)

        # Initial reset with longer timeout
        logging.info("Performing initial reset...")
        observations, _ = env.reset()
        time.sleep(2)  # Add delay after reset

        for step in range(1, num_steps + 1):
            step_start = time.time()
            
            # Select random actions for each client
            actions = [random.randint(0, env.action_space.n - 1) for _ in range(env.num_clients)]
            action_timestamp = datetime.now().isoformat()
            
            try:
                # Perform step
                observations, rewards, terminated, dones, infos = env.step(actions)
                step_duration = time.time() - step_start
                received_timestamp = datetime.now().isoformat()

                # Log actions and observations
                for i in range(env.num_clients):
                    if isinstance(observations, dict):  # Handle single client case
                        obs_to_log = observations
                    else:
                        obs_to_log = observations[i]
                    
                    serialized_obs = serialize_observation(obs_to_log)
                    log_entry = {
                        "Step": step,
                        "Timestamp_Action_Sent": action_timestamp,
                        "Action": actions[i],
                        "Action_Name": env.ACTION_MAPPING[actions[i]],
                        "Timestamp_Received": received_timestamp,
                        "Duration_sec": step_duration,
                        "Reward": rewards[i] if isinstance(rewards, list) else rewards,
                        "Done": dones[i] if isinstance(dones, list) else dones,
                        "Info": infos[i] if isinstance(infos, list) else infos,
                        "Observation": serialized_obs
                    }
                    log_files[i].write(json.dumps(log_entry) + "\n")
                    log_files[i].flush()

                logging.info(f"Step {step}/{num_steps} completed in {step_duration*1000:.2f}ms")
                
                # Check if all clients are done
                if isinstance(dones, list) and all(dones):
                    logging.info("All clients finished. Ending episode.")
                    break
                elif isinstance(dones, bool) and dones:
                    logging.info("Client finished. Ending episode.")
                    break

                # Add delay between steps
                time.sleep(0.1)

            except Exception as e:
                logging.error(f"Error during step {step}: {e}")
                break

    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        # Print full traceback for debugging
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # Close log files if they were opened
        if log_files:
            for f in log_files:
                try:
                    f.close()
                except Exception as e:
                    logging.error(f"Error closing log file: {e}")
        
        # Close environment if it was created
        try:
            if 'env' in locals():
                env.close()
        except Exception as e:
            logging.error(f"Error closing environment: {e}")
        
        logging.info("Test script completed.")

if __name__ == "__main__":
    main()
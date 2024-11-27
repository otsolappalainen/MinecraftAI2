import asyncio
import websockets
import json
import logging
from datetime import datetime

# Configuration Constants
WS_URI = "ws://localhost:8080"  # WebSocket URI of the Minecraft mod
ACTION_SPACE_SIZE = 20          # Total number of actions (0-19)
TOTAL_ACTIONS = 20               # Number of actions to send in the test
DEBUG = True                     # Toggle debug logging

# Action Mapping based on the Minecraft Gym environment
ACTION_MAPPING = {
    0: "move_forward",
    1: "move_backward",
    2: "move_left",
    3: "move_right",
    4: "jump_walk_forward",
    5: "jump",
    6: "sneak_toggle",
    7: "look_left",
    8: "look_right",
    9: "look_up",
    10: "look_down",
    11: "big_turn_left",
    12: "big_turn_right",
    13: "mouse_left_click",
    14: "mouse_right_click",
    15: "next_item",
    16: "previous_item",
    17: "reset 0",
    18: "reset 1",
    19: "reset 2",
}

# Define the sequence of actions (mostly walking)
ACTION_SEQUENCE = [
    0, 0, 0, 0,  # move_forward
    1, 1,        # move_backward
    2, 2,        # move_left
    3, 3,        # move_right
    0,           # move_forward
    1,           # move_backward
    2,           # move_left
    3,           # move_right
    4,           # jump_walk_forward
    5,           # jump
    7, 8,        # look_left, look_right
    0            # move_forward
]

# Initialize Logging
def init_logging(debug: bool):
    logger = logging.getLogger("TestModEnv")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# Create the logger
logger = init_logging(DEBUG)

async def test_mod_env():
    """
    Test function to send actions to the Minecraft mod and receive state updates.
    Logs the time each action is sent and when the response is received.
    """
    try:
        async with websockets.connect(WS_URI) as websocket:
            logger.info(f"Connected to WebSocket server at {WS_URI}")
            
            for idx, action_idx in enumerate(ACTION_SEQUENCE):
                if action_idx >= ACTION_SPACE_SIZE:
                    logger.warning(f"Action index {action_idx} is out of bounds. Skipping.")
                    continue

                action_name = ACTION_MAPPING.get(action_idx, "no_op")
                
                # Prepare the action message
                action_message = json.dumps({"action": action_name})
                
                # Record the send time
                send_time = datetime.now()
                logger.debug(f"Sending Action {idx+1}/{TOTAL_ACTIONS}: '{action_name}' at {send_time}")
                
                # Send the action
                await websocket.send(action_message)
                
                # Await the response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    recv_time = datetime.now()
                    logger.debug(f"Received Response for Action '{action_name}' at {recv_time}")
                    
                    # Log the time difference
                    time_diff = (recv_time - send_time).total_seconds()
                    logger.info(f"Action '{action_name}' sent at {send_time.strftime('%H:%M:%S.%f')[:-3]}, "
                                f"received at {recv_time.strftime('%H:%M:%S.%f')[:-3]} "
                                f"(Duration: {time_diff:.3f} seconds)")
                    
                    # Optionally, log the response content
                    logger.debug(f"Response Content: {response}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Timeout: No response received for action '{action_name}' within 5 seconds.")
                
                # Optional: Wait a short duration before sending the next action
                await asyncio.sleep(0.1)  # 100 milliseconds

            logger.info("Completed sending all actions.")
    
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    asyncio.run(test_mod_env())
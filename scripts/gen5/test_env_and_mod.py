import websocket
import json
import time
import csv
import threading
import keyboard

# WebSocket URL
WEBSOCKET_URL = "ws://localhost:8080"

# Actions to be tested
actions = [
    {"action": "reset", "reset_type": 1, "coordinates": {"x": 0.0, "y": 64.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}},
    {"action": "move_forward"},
    {"action": "turn_left"},
    {"action": "move_right"},
    {"action": "reset", "reset_type": 2, "coordinates": {"x": 0.0, "y": 64.0, "z": 0.0}},
]

# Log file
LOG_FILE = "test_script_log.csv"

# Function to connect to the WebSocket server and send actions
def run_test_script():
    try:
        # Establish WebSocket connection
        ws = websocket.WebSocket()
        ws.connect(WEBSOCKET_URL)
        
        # Open log file
        with open(LOG_FILE, mode="w", newline="") as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(["action", "send_timestamp", "response_timestamp", "response"])
            
            for i in range(20):
                action = actions[i % len(actions)]
                # Send action
                send_timestamp = time.time()
                ws.send(json.dumps(action))
                print(f"Sent: {json.dumps(action)}")
                
                # Wait for response
                response = ws.recv()
                response_timestamp = time.time()
                print(f"Received: {response}")
                
                # Log the action, send and response timestamps, and response
                log_writer.writerow([json.dumps(action), send_timestamp, response_timestamp, response])
        
        # Close the WebSocket connection
        ws.close()
    except Exception as e:
        print(f"Error: {e}")

# Function to wait for key press to start the test
def wait_for_keypress():
    print("Press 'o' to start the test script...")
    keyboard.wait('o')
    run_test_script()

# Run the test script if 'o' is pressed
if __name__ == "__main__":
    keypress_thread = threading.Thread(target=wait_for_keypress)
    keypress_thread.start()
    keypress_thread.join()

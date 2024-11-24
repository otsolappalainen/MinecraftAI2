import websocket
import json
import time
import signal
from threading import Thread, Event
from pynput import keyboard

# Define WebSocket connection URL
WS_URL = "ws://localhost:8080"

# Actions to test
ACTIONS = [
    {"action": "move_forward"},
    {"action": "move_backward"},
    {"action": "jump"},
    {"action": "sneak_toggle"},
    {"action": "mouse_left_click"},  # Toggle left click
    {"action": "mouse_right_click"},  # Right-click once
    {"action": "look_left"},
    {"action": "look_right"},
    {"action": "look_up"},
    {"action": "look_down"},
    {"action": "big_turn_left"},
    {"action": "big_turn_right"},
    {"action": "next_item"},  # Cycle to the next item in the hotbar
    {"action": "previous_item"},  # Cycle to the previous item in the hotbar
]

# Delay between repeated actions (in seconds)
ACTION_DELAY = 2
REPEAT_COUNT = 4  # Number of times to repeat each action

# Start and stop events
start_event = Event()
stop_event = Event()


def wait_for_start():
    """
    Wait for the user to press the 'o' key to start the script.
    """
    def on_press(key):
        try:
            if key.char == "o":  # Wait for 'o' key
                print("Starting the script...")
                start_event.set()
                return False  # Stop listening
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def on_message(ws, message):
    """
    Callback when a message is received from the server.
    """
    try:
        data = json.loads(message)
        print(f"Received data: {json.dumps(data, indent=2)}")
    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")


def on_error(ws, error):
    """
    Callback when an error occurs.
    """
    print(f"Error: {error}")


def on_close(ws, close_status_code, close_msg):
    """
    Callback when the connection is closed.
    """
    print("WebSocket connection closed.")


def on_open(ws):
    """
    Callback when the connection is opened.
    """
    print("WebSocket connection opened.")

    def send_actions():
        # Send each action 4 times in a row
        for action in ACTIONS:
            for _ in range(REPEAT_COUNT):
                if stop_event.is_set():  # Exit if stop_event is triggered
                    return
                print(f"Sending action: {action}")
                try:
                    ws.send(json.dumps(action))  # Send the action as JSON
                except Exception as e:
                    print(f"Error sending action {action}: {e}")
                time.sleep(ACTION_DELAY)  # Wait for the action to process

    Thread(target=send_actions, daemon=True).start()


def handle_termination(signum, frame):
    """
    Handle termination signals (e.g., Ctrl+C).
    """
    print("Termination signal received. Stopping the script...")
    stop_event.set()  # Trigger the stop event


def main():
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)

    # Start a thread to listen for the 'o' key
    Thread(target=wait_for_start, daemon=True).start()

    # Wait until 'o' key is pressed
    print("Waiting for the 'o' key to start...")
    start_event.wait()

    # Create a WebSocket connection
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # Run the WebSocket connection
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Closing WebSocket...")
        stop_event.set()  # Trigger the stop event


if __name__ == "__main__":
    main()

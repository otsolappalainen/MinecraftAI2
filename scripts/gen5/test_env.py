import time
from stable_baselines3 import PPO
from env import SimulatedEnvGraphics
from threading import Thread, Event
from pynput import keyboard

# Event to signal when to start
start_event = Event()

def wait_for_start():
    """Wait for the user to press the 'o' key to start the script."""
    def on_press(key):
        try:
            if key.char == "o":  # Wait for the 'o' key
                print("Starting the environment...")
                start_event.set()  # Signal to start
                return False  # Stop listening
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Start the key listener in a separate thread
listener_thread = Thread(target=wait_for_start, daemon=True)
listener_thread.start()

print("Waiting for the 'o' key to start...")
start_event.wait()  # Block until the 'o' key is pressed

# Initialize the environment and run the test
env = SimulatedEnvGraphics()
env.reset()

try:
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step reward: {reward}")
        if done:
            break
finally:
    env.close()

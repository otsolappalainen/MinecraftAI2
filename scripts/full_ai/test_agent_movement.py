import time
import keyboard  # Requires 'keyboard' library for background key listening
import threading
from agent import MinecraftAgent  # Ensure this is the correct path to your MinecraftAgent

class AgentMovementTester:
    def __init__(self, agent):
        self.agent = agent
        self.running = False  # Flag to control the test loop
        self.test_thread = None  # Thread for running movements
        self.listen_thread = None  # Thread for listening to stop key

        # Register the hotkeys
        keyboard.add_hotkey('o', self.start_test)
        keyboard.add_hotkey('i', self.stop_test)

    def start_test(self):
        """Start the movement test."""
        if not self.running:
            print("Starting movement test...")
            self.running = True
            self.test_thread = threading.Thread(target=self.run_movements)
            self.test_thread.start()
            self.listen_thread = threading.Thread(target=self.listen_for_stop)
            self.listen_thread.start()

    def stop_test(self):
        """Stop the movement test gracefully."""
        print("Stopping movement test...")
        self.running = False
        if self.test_thread and self.test_thread.is_alive():
            self.test_thread.join()
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join()

    def listen_for_stop(self):
        """Listen for the stop key ('I') in the background."""
        while self.running:
            if keyboard.is_pressed('i'):
                self.stop_test()
                break
            time.sleep(0.1)

    def run_movements(self):
        """Run a sequence of movements with varied durations."""
        movements = [
            ('move_forward', 1.0),
            ('move_backward', 0.5),
            ('strafe_left', 1.5),
            ('strafe_right', 1.0),
            ('turn_left', 1.0),
            ('turn_right', 0.5),
            ('look_up', 0.75),
            ('look_down', 1.0),
            ('jump', 0.2),
            ('toggle_sneak', 1.5)
        ]

        while self.running:
            for action, duration in movements:
                print(f"Executing {action} for {duration} seconds...")
                
                # Ensure no other movement is active
                self.agent.stop_all_movements()
                
                # Execute the action
                getattr(self.agent, action)()

                # Wait for the specified duration
                time.sleep(duration)

                # Stop the action if it has a stop method
                if hasattr(self.agent, f"stop_{action}"):
                    getattr(self.agent, f"stop_{action}")()

                if not self.running:
                    break

        print("Movement test completed.")

if __name__ == "__main__":
    agent = MinecraftAgent()
    tester = AgentMovementTester(agent)
    
    print("Press 'O' to start the movement test and 'I' to stop it.")
    keyboard.wait('esc')


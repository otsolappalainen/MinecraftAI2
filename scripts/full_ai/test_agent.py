import cv2
import numpy as np
import time
import keyboard  # Requires 'keyboard' library
from agent import MinecraftAgent  # Make sure to replace with the actual path of your agent

class AgentTester:
    def __init__(self, agent, mode="check_image_and_coordinates"):
        self.agent = agent
        self.mode = mode
        self.running = True  # Flag to control the loop
        self.process_times = []  # List to store total process times per iteration

        # Dictionary to store individual timers
        self.timers = {
            "get_state": [],
            "capture_full_screen": [],
            "process_image": [],
            "display_frame": []
        }

        # Set up the kill switch listener
        keyboard.add_hotkey('esc', self.stop_program)

    def stop_program(self):
        """Stop the main loop and close the program gracefully."""
        print("ESC pressed. Exiting the program.")
        self.running = False

    def display_state(self):
        """
        Display the agent's coordinates, angles, and screenshots in real-time.
        """
        cv2.namedWindow("Agent State Display", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Agent State Display", 800, 600)

        while self.running:
            # Start main timer for the entire loop
            loop_start_time = time.time()

            # Start timing for get_state
            start_time = time.time()
            self.agent.get_state()
            self.timers["get_state"].append(time.time() - start_time)


            

            # Extract agent's state
            x, y, z, yaw, pitch, processed_image = self.agent.state

            # Ensure valid values by using previous state if any are invalid
            if x is None or y is None or z is None or yaw is None or pitch is None:
                x, y, z, yaw, pitch, processed_image = self.agent.previous_state

            # Ensure we have a valid processed_image
            if processed_image is None:
                processed_image = np.zeros((224, 224), dtype=np.uint8)
            else:
                # Convert tensor to NumPy array
                processed_image = processed_image.squeeze().cpu().numpy()

            # Start timing for display processing
            start_time = time.time()
            
            display_image = cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(display_image, f"X: {x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_image, f"Y: {y:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_image, f"Z: {z:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_image, f"Yaw: {yaw:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_image, f"Pitch: {pitch:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            self.timers["display_frame"].append(time.time() - start_time)  # End timing for display processing

            # Display the image in the window
            cv2.imshow("Agent State Display", display_image)

            # Record the total time for this iteration
            loop_time = time.time() - loop_start_time
            self.process_times.append(loop_time)

            # Print timing details
            #print(f"\nFull frame processing time: {loop_time:.4f} seconds")
            #print(f"get_state execution time: {self.timers['get_state'][-1]:.4f} seconds")
            #print(f"display_frame execution time: {self.timers['display_frame'][-1]:.4f} seconds")

            # Close window on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def run_test(self):
        if self.mode == "check_image_and_coordinates":
            print("Running Agent Test: Check Image and Coordinates")
            self.display_state()
        elif self.mode == "test_outputs":
            print("Running Agent Test: Test Outputs")
            self.test_outputs()
        else:
            print("Invalid mode selected.")

        # Remove the keyboard hook when done
        keyboard.remove_hotkey('esc')

        # Print summary of average times after the test
        print("\n--- Average Execution Times ---")
        if self.process_times:
            avg_total_time = sum(self.process_times) / len(self.process_times)
            print(f"Average full frame processing time: {avg_total_time:.4f} seconds")

        for func, times in self.timers.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"{func} average execution time: {avg_time:.4f} seconds")


if __name__ == "__main__":
    # Initialize the agent
    agent = MinecraftAgent()
    
    # Initialize the tester in "check_image_and_coordinates" mode
    tester = AgentTester(agent, mode="check_image_and_coordinates")
    
    # Run the test
    tester.run_test()

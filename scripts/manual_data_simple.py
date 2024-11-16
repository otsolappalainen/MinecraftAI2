import cv2
import numpy as np
import os
from datetime import datetime
from mss import mss
from pynput import mouse

# Set up screen capture region and save directory
screen_region = {
    "top": int((2160 - 720) / 2),      # Adjust based on your screen resolution
    "left": int((3840 - 1280) / 2),    # Adjust based on your screen resolution
    "width": 1280,
    "height": 720
}
save_dir = r"C:/Users/odezz/source/MinecraftAI2/screenshots"  # Change this path as needed
os.makedirs(save_dir, exist_ok=True)

# Initialize screen capture
sct = mss()

# Flag to control capture
capture_flag = False

# Function to capture and save a screenshot
def capture_screenshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(image_path, frame)
    print(f"Captured: {image_path}")

# Mouse event handler
def on_click(x, y, button, pressed):
    global capture_flag
    if button == mouse.Button.x2 and pressed:  # Check for mouse4 press
        capture_flag = True

# Set up mouse listener
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

# Main loop for live preview and capture
try:
    print("Press mouse4 to capture a screenshot. Press 'q' to exit.")
    while True:
        # Capture the screen region and display the preview
        screenshot = np.array(sct.grab(screen_region))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # Display the live preview in an OpenCV window
        cv2.imshow("Screen Preview", frame)
        
        # Check if the capture flag is set
        if capture_flag:
            capture_screenshot(frame)
            capture_flag = False  # Reset capture flag after each capture

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("\nExiting program.")

# Stop the mouse listener and close OpenCV windows
mouse_listener.stop()
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from PIL import ImageGrab
import time
from pynput import mouse

# Directory to save screenshots
IMAGE_DIR = "minecraft_screenshots"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Capture region configuration
CAPTURE_REGION = {"x_offset": 519, "y_offset": 385, "width": 276, "height": 31}

def capture_image():
    """Capture a defined region from the screen and return it as an image."""
    x, y, w, h = CAPTURE_REGION['x_offset'], CAPTURE_REGION['y_offset'], CAPTURE_REGION['width'], CAPTURE_REGION['height']
    bbox = (x, y, x + w, y + h)
    screenshot = ImageGrab.grab(bbox=bbox)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def save_image(img):
    """Save the image to file."""
    timestamp = int(time.time() * 1000)
    image_path = os.path.join(IMAGE_DIR, f"{timestamp}.png")
    cv2.imwrite(image_path, img)
    print(f"Saved image: {image_path}")

def on_click(x, y, button, pressed):
    """Handle mouse click events to save image on Button.x2 press."""
    if button == mouse.Button.x2 and pressed:
        img = capture_image()
        save_image(img)

def main():
    # Start the mouse listener for Button.x2 to save images
    print("Press the side mouse button (Button.x2) to capture images. Press 'q' in the preview window to quit.")
    
    with mouse.Listener(on_click=on_click) as listener:
        last_capture_time = 0
        
        while True:
            # Only capture and update the preview every second
            current_time = time.time()
            if current_time - last_capture_time >= 1:
                img = capture_image()
                cv2.imshow("Captured Region", img)
                last_capture_time = current_time

            # Check for 'q' key press to exit
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("Stopping capture.")
                break

        listener.stop()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


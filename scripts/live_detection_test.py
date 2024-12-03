import cv2
import numpy as np
#from ultralytics import YOLO
from mss import mss

# Load YOLOv8 model in inference mode
model_path = r"C:\Users\odezz\source\MinecraftAI2\runs\exp13\weights\best.pt"
#model = YOLO(model_path)

# Define the screen capture region (1280x720 centered on a 4K screen)
monitor = {
    "top": (2160 - 720) // 2,   # Center vertically on 4K (2160p) screen
    "left": (3840 - 1280) // 2, # Center horizontally on 4K (3840p) screen
    "width": 1280,
    "height": 720
}

# Initialize mss for screen capture
sct = mss()

def capture_screen():
    """Capture a 1280x720 region of the screen."""
    sct_img = sct.grab(monitor)
    img = np.array(sct_img)
    # Convert BGRA to BGR (remove alpha channel) if needed
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def main():
    try:
        while True:
            # Capture the screen
            frame = capture_screen()

            # Perform inference with YOLOv8
            #results = model.predict(source=frame, conf=0.5)

            # Annotate frame with bounding boxes and labels
            #annotated_frame = results[0].plot()  # Plot detections on the frame

            # Show the annotated frame
            #cv2.imshow("YOLOv8 Live Detection", annotated_frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up and close the display window
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
from pynput import mouse
from datetime import datetime
import os

# Load YOLOv8 model
model_path = r"C:\Users\odezz\source\MinecraftAI2\runs\exp13\weights\best.pt"
model = YOLO(model_path)

# Screen capture region (1280x720 centered on a 4K screen)
monitor = {
    "top": (2160 - 720) // 2,
    "left": (3840 - 1280) // 2,
    "width": 1280,
    "height": 720
}

# Initialize mss for screen capture
sct = mss()
frame_counter = 0
capture_mode = False  # Start without capturing images

# Set up directories for saving images and labels
save_dir = "captured_data"
os.makedirs(save_dir, exist_ok=True)

def capture_screen():
    """Capture a 1280x720 region of the screen."""
    sct_img = sct.grab(monitor)
    img = np.array(sct_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def on_click(x, y, button, pressed):
    """Toggle capture mode when Mouse4 (Button.x2) is pressed."""
    global capture_mode
    if button == mouse.Button.x2 and pressed:
        capture_mode = not capture_mode
        print(f"Capture mode {'enabled' if capture_mode else 'disabled'}.")

# Start a listener for Mouse4 (Button.x2)
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

def save_detection(image, detections):
    """Save an image and its YOLO-format labels without overlaying bounding boxes."""
    img_h, img_w = image.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(save_dir, f"image_{timestamp}.png")
    label_path = os.path.join(save_dir, f"image_{timestamp}.txt")

    # Save the original image
    cv2.imwrite(image_path, image)

    # Write labels in YOLO format
    with open(label_path, 'w') as f:
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0]
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            class_id = int(box.cls[0])
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Saved: {image_path}, {label_path}")

def main():
    global frame_counter
    try:
        while True:
            frame = capture_screen()
            results = model.predict(source=frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()  # Plot detections for display

            # Display the annotated frame
            cv2.imshow("YOLOv8 Live Detection", annotated_frame)

            # Capture images every 10 frames if capture mode is active
            if capture_mode and frame_counter % 10 == 0:
                detections = results[0].boxes  # Access detection boxes
                save_detection(frame, detections)

            # Increment the frame counter
            frame_counter += 1

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the mouse listener and close display windows
        mouse_listener.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

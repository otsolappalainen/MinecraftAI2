import cv2
import numpy as np
import os
from datetime import datetime
from mss import mss
from pynput import mouse
import keyboard
import threading  # Import threading to use for keyboard listener

# Set up screen capture region and save directory
screen_region = {
    "top": int((2160 - 720) / 2),      # Centered vertically on a 4K screen
    "left": int((3840 - 1280) / 2),    # Centered horizontally on a 4K screen
    "width": 1280,
    "height": 720
}
save_dir = r"C:/Users/odezz/source/MinecraftAI2/yolov8/manual_data"
os.makedirs(save_dir, exist_ok=True)

# Initialize screen capture
sct = mss()

# Bounding box parameters
available_classes = ['valuable block', 'torch', 'lava', 'water']
current_class_index = 0

# List to store bounding boxes in the form: [(x, y, width, height, class_index)]
bounding_boxes = []
selected_box_index = -1  # Index of the currently selected bounding box
capture_flag = False


# Function to save the current image and all bounding boxes in YOLO format
def save_capture(image, boxes):
    img_h, img_w = image.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(save_dir, f"image_{timestamp}.png")
    label_path = os.path.join(save_dir, f"image_{timestamp}.txt")

    # Save the image without the green boxes
    cv2.imwrite(image_path, image)

    # Write all bounding boxes to the text file in YOLO format
    with open(label_path, 'w') as f:
        for (x, y, w, h, class_index) in boxes:
            # Convert coordinates to YOLO format
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            yolo_width = w / img_w
            yolo_height = h / img_h
            f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")
    
    print(f"Saved: {image_path} and {label_path}")

# Trackbar callback function (does nothing but is required for trackbar)
def nothing(x):
    pass

# Mouse control for capturing
def on_click(x, y, button, pressed):
    global capture_flag
    if button == mouse.Button.x2 and pressed:
        capture_flag = True

# Initialize mouse listener
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

# Create a window with trackbars for rectangle adjustment
cv2.namedWindow("Manual Labeling Tool")
cv2.createTrackbar("X Position", "Manual Labeling Tool", 0, screen_region['width'] - 10, nothing)
cv2.createTrackbar("Y Position", "Manual Labeling Tool", 0, screen_region['height'] - 10, nothing)
cv2.createTrackbar("Width", "Manual Labeling Tool", 100, screen_region['width'], nothing)
cv2.createTrackbar("Height", "Manual Labeling Tool", 100, screen_region['height'], nothing)
cv2.createTrackbar("Class", "Manual Labeling Tool", current_class_index, len(available_classes) - 1, nothing)

# Functions to handle box events
def add_new_box():
    global selected_box_index
    bounding_boxes.append((50, 50, 100, 100, current_class_index))
    selected_box_index = len(bounding_boxes) - 1
    print(f"Added new box. Total boxes: {len(bounding_boxes)}")

def select_next_box():
    global selected_box_index
    if bounding_boxes:
        selected_box_index = (selected_box_index + 1) % len(bounding_boxes)
        print(f"Selected box {selected_box_index + 1} of {len(bounding_boxes)}")

def delete_selected_box():
    global selected_box_index
    if selected_box_index != -1 and bounding_boxes:
        print(f"Deleted box {selected_box_index + 1}")
        bounding_boxes.pop(selected_box_index)
        selected_box_index = max(0, selected_box_index - 1) if bounding_boxes else -1

def increase_box_size():
    if selected_box_index != -1:
        x, y, w, h, class_index = bounding_boxes[selected_box_index]
        bounding_boxes[selected_box_index] = (x, y, w + 5, h + 5, class_index)

def decrease_box_size():
    if selected_box_index != -1:
        x, y, w, h, class_index = bounding_boxes[selected_box_index]
        bounding_boxes[selected_box_index] = (x, y, max(w - 5, 5), max(h - 5, 5), class_index)

def move_box(dx, dy):
    global bounding_boxes
    if selected_box_index != -1 and 0 <= selected_box_index < len(bounding_boxes):
        x, y, w, h, class_index = bounding_boxes[selected_box_index]
        
        new_x = max(0, min(x + dx, screen_region["width"] - w))
        new_y = max(0, min(y + dy, screen_region["height"] - h))
        
        bounding_boxes[selected_box_index] = (new_x, new_y, w, h, class_index)
        print(f"Moved box to new position: ({new_x}, {new_y})")  # Debug print to confirm position change

# Keyboard handling function for global hotkeys in a separate thread
def keyboard_listener():
    keyboard.add_hotkey('n', add_new_box)
    keyboard.add_hotkey('m', select_next_box)
    keyboard.add_hotkey('b', delete_selected_box)
    keyboard.add_hotkey('i', increase_box_size)
    keyboard.add_hotkey('o', decrease_box_size)
    keyboard.add_hotkey('u', lambda: move_box(0, -5))  # Move up
    keyboard.add_hotkey('j', lambda: move_box(0, 5))   # Move down
    keyboard.add_hotkey('h', lambda: move_box(-5, 0))  # Move left
    keyboard.add_hotkey('k', lambda: move_box(5, 0))   # Move right
    keyboard.wait()  # Keeps the listener thread alive

# Start the keyboard listener in a separate thread
keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
keyboard_thread.start()

# Main loop to capture screen and display rectangles
while True:
    # Capture the screen
    screen_capture = np.array(sct.grab(screen_region))
    frame = cv2.cvtColor(screen_capture, cv2.COLOR_BGRA2BGR)  # Convert the captured image to BGR for OpenCV
    
    # Create a copy for display
    display_frame = frame.copy()

    # Draw all bounding boxes on the display frame
    for i, (x, y, w, h, class_index) in enumerate(bounding_boxes):
        color = (0, 255, 0) if i == selected_box_index else (255, 0, 0)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
        label_text = f"Class: {available_classes[class_index]}"
        cv2.putText(display_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the display frame with the green boxes
    cv2.imshow("Manual Labeling Tool", display_frame)

    # Check if capture is requested
    if capture_flag:
        save_capture(frame, bounding_boxes)
        capture_flag = False

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop listeners and release resources
mouse_listener.stop()
keyboard.unhook_all_hotkeys()
cv2.destroyAllWindows()

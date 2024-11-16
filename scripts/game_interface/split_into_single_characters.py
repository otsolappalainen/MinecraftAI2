import cv2
import os
import numpy as np
from pynput import keyboard
import time

# Directory paths
image_dir = "minecraft_screenshots"       # Folder with the original screenshots
output_dir = "cropped_digits"             # Folder where labeled images will be saved
os.makedirs(output_dir, exist_ok=True)

# Create folders for each character class
label_classes = "0123456789-/. "
for label in label_classes:
    if label == " ":
        os.makedirs(os.path.join(output_dir, "space"), exist_ok=True)
    elif label == ".":
        os.makedirs(os.path.join(output_dir, "dot"), exist_ok=True)
    elif label == "/":
        os.makedirs(os.path.join(output_dir, "slash"), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Adjustable parameters
window_width = 20  # Width of the sliding window, adjust if needed
window_height = 30  # Height of the sliding window
step_size = 2       # Step size for sliding the window

# Initialize variables
current_x = 0  # Starting x position for sliding window
current_y = 0
current_image = None
current_image_path = None
exit_program = False  # Flag to indicate if the program should exit

def save_cropped_character(label):
    """Crop and save the character image based on the current window position and label."""
    global current_image, current_x, current_y
    cropped_char = current_image[current_y:current_y+window_height, current_x:current_x+window_width]
    label_folder = "space" if label == " " else ("dot" if label == "." else ("slash" if label == "/" else label))
    save_path = os.path.join(output_dir, label_folder, f"{int(time.time() * 1000)}.png")
    cv2.imwrite(save_path, cropped_char)
    print(f"Saved '{label}' at {save_path}")

def on_key_press(key):
    """Handle key press events to control window movement and labeling."""
    global current_x, current_y, exit_program

    try:
        if key == keyboard.Key.right:
            current_x = min(current_x + step_size, current_image.shape[1] - window_width)
        elif key == keyboard.Key.left:
            current_x = max(current_x - step_size, 0)
        elif key == keyboard.Key.down:
            current_y = min(current_y + step_size, current_image.shape[0] - window_height)
        elif key == keyboard.Key.up:
            current_y = max(current_y - step_size, 0)
        elif hasattr(key, 'char') and key.char in label_classes:
            save_cropped_character(key.char)
        elif key == keyboard.Key.esc:
            # Set flag to exit the current image
            exit_program = True
            return False  # Stop listener for this image
    except AttributeError:
        pass  # Ignore other keys

    return True

def process_images():
    """Load each image and display it with the sliding window."""
    global current_image, current_image_path, current_x, current_y, exit_program

    for filename in sorted(os.listdir(image_dir)):
        current_image_path = os.path.join(image_dir, filename)
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        
        if current_image is None:
            print(f"Could not load image: {filename}")
            continue

        current_x, current_y = 0, 0  # Reset window position for each image
        exit_program = False  # Reset flag for each image

        # Start the key listener for labeling
        with keyboard.Listener(on_press=on_key_press) as listener:
            while True:
                # Display the image with the sliding window
                display_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(display_image, (current_x, current_y), 
                              (current_x + window_width, current_y + window_height), 
                              (0, 255, 0), 2)
                cv2.imshow("Character Labeling", display_image)

                # Check if we need to close the current image or the whole program
                if exit_program:
                    listener.stop()  # Stop listening to keys for this image
                    break

                # Wait for a short time and check if window is closed
                if cv2.waitKey(100) == 27:  # 'ESC' key to exit
                    exit_program = True
                    break

                if cv2.getWindowProperty("Character Labeling", cv2.WND_PROP_VISIBLE) < 1:
                    break  # Break if window is closed manually

            if exit_program:
                break  # Exit the program if `ESC` was pressed

    print("Finished processing all images.")

# Setup and run the labeling process
print("Starting character labeling. Use arrow keys to position the window, press 0-9, '-', '/', '.', or space to label, and ESC to skip the current image or exit the program.")
process_images()

# Cleanup
cv2.destroyAllWindows()

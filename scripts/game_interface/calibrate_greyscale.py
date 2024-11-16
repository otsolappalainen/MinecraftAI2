import cv2
import numpy as np
import os
import json

# Directory path
image_dir = "minecraft_screenshots"

# Default threshold
default_threshold = 128  # Default binary cutoff (0 to 255 range)

# Load an image from the directory
def load_image():
    filenames = sorted(os.listdir(image_dir))
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    print("No valid image found in the directory.")
    return None

# Function to apply binary threshold transformation
def apply_binary_threshold(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# Trackbar callback function for threshold adjustment
def on_threshold_change(value):
    global threshold
    threshold = value

def main():
    global threshold

    # Initialize default threshold value
    threshold = default_threshold

    # Load an image
    image = load_image()
    if image is None:
        print("No images to display.")
        return

    # Create a window and trackbar for manual threshold adjustment
    cv2.namedWindow("Binary Threshold Calibration", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", "Binary Threshold Calibration", threshold, 255, on_threshold_change)

    print("Adjust the binary threshold using the slider. Press 's' to save the parameter or 'q' to quit.")

    while True:
        # Apply binary threshold with the current threshold value
        transformed_image = apply_binary_threshold(image, threshold)

        # Display the transformed image
        cv2.imshow("Binary Threshold Calibration", transformed_image)

        # Wait for key press
        key = cv2.waitKey(10)
        if key == ord('s'):
            # Save selected parameter to a config file
            with open("binary_threshold_config.json", "w") as f:
                config = {
                    "threshold": threshold
                }
                json.dump(config, f)
            print(f"Parameter saved: Binary Threshold = {threshold}")
            break
        elif key == ord('q') or cv2.getWindowProperty("Binary Threshold Calibration", cv2.WND_PROP_VISIBLE) < 1:
            # Exit without saving
            print("Calibration exited without saving parameter.")
            break

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


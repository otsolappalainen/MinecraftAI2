from PIL import ImageGrab
import cv2
import numpy as np

# Define the regions for testing
COORD_REGION = {"x_offset": 66, "y_offset": 308, "width": 530, "height": 29}

def test_capture_region(region):
    bbox = (region["x_offset"], region["y_offset"],
            region["x_offset"] + region["width"],
            region["y_offset"] + region["height"])
    screenshot = ImageGrab.grab(bbox=bbox).convert("L")  # Convert to grayscale
    return np.array(screenshot)

# Capture and display the region
image = test_capture_region(COORD_REGION)
cv2.imshow("Test Capture", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
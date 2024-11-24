import time
import numpy as np
import cv2
from mss import mss
from PIL import Image

# Define capture and processing methods
def capture_mss(region):
    """Capture screen using MSS."""
    with mss() as sct:
        screenshot = sct.grab(region)
        return np.array(screenshot)

def process_resize_opencv(image, size=(224, 224)):
    """Process image by resizing using OpenCV."""
    return cv2.resize(image, size)

def process_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def capture_dxgi(region, output_resolution=(224, 224)):
    """Capture screen using DXGI and resize using OpenCV."""
    with mss() as sct:
        img = sct.grab(region)
        frame = np.array(img)
        return cv2.resize(frame, output_resolution, interpolation=cv2.INTER_LINEAR)

# Modular pipeline
def capture_and_process(capture_method, process_methods, region, size=(224, 224)):
    """
    Capture the screen using `capture_method`, and apply a sequence of `process_methods`.
    
    Args:
        capture_method: Function to capture the screen.
        process_methods: List of functions to process the image.
        region: Region of interest for capture.
        size: Desired output size of the processed image.
    
    Returns:
        Processed image.
    """
    # Step 1: Capture the image
    image = capture_method(region)
    
    # Step 2: Process the image through each processing step
    for process in process_methods:
        image = process(image)
    
    return image

# Measure FPS
def measure_fps(capture_method, process_methods, region, iterations=100):
    """
    Measure FPS for a given capture and processing pipeline.
    
    Args:
        capture_method: Function to capture the screen.
        process_methods: List of functions to process the image.
        region: Region of interest for capture.
        iterations: Number of iterations to measure.
    """
    start_time = time.time()
    
    for _ in range(iterations):
        capture_and_process(capture_method, process_methods, region)
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = iterations / total_time
    print(f"Processed {iterations} frames in {total_time:.2f} seconds. FPS: {fps:.2f}")

# Define the region of interest (example region)
region = {
    "top": 0,            # Starting Y-coordinate (center of 4K display)
    "left": 840,           # Starting X-coordinate (center of 4K display)
    "width": 720,         # Width of the ROI
    "height": 720         # Height of the ROI
}

# Test configurations
if __name__ == "__main__":
    print(f"Region of Interest: {region}")

    # Test 1: MSS + OpenCV Resize
    print("Test 1: MSS + OpenCV Resize")
    measure_fps(capture_mss, [lambda x: process_resize_opencv(x, size=(224, 224))], region)
    
    # Test 2: MSS + Grayscale
    print("\nTest 2: MSS + Grayscale + Resize")
    measure_fps(capture_mss, [process_grayscale, lambda x: process_resize_opencv(x, size=(224, 224))], region)
    
    # Test 3: MSS without processing
    print("\nTest 3: MSS Without Processing")
    measure_fps(capture_mss, [], region)

    # Test 4: DXGI Capture + OpenCV Resize
    print("\nTest 4: DXGI Capture + OpenCV Resize")
    measure_fps(lambda region: capture_dxgi(region, output_resolution=(224, 224)), [], region)

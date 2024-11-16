import cv2
import numpy as np
import pytesseract
import time
import json
import os
import pygetwindow as gw
import win32gui
from PIL import ImageGrab
from threading import Thread, Event
from pynput import keyboard
import re

# Configuration file to store region coordinates
CONFIG_FILE = 'region_config.json'

expected_ranges = {
    "x": (-400, -200),  
    "y": (-70, -20),
    "z": (300, 500),
}

# Event to signal threads to stop
stop_event = Event()

def save_region_config(region):
    with open(CONFIG_FILE, 'w') as config_file:
        json.dump(region, config_file)

def load_region_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as config_file:
            return json.load(config_file)
    return {'x_offset': 0, 'y_offset': 0, 'width': 300, 'height': 200}

def find_minecraft_window():
    windows = gw.getAllTitles()
    for window in windows:
        if "24w45" in window and "Minecraft Launcher" not in window:
            hwnd = win32gui.FindWindow(None, window)
            if hwnd:
                return hwnd
    return None

def capture_region(region, hwnd):
    x, y, width, height = region['x_offset'], region['y_offset'], region['width'], region['height']
    win_rect = win32gui.GetWindowRect(hwnd)
    bbox = (win_rect[0] + x, win_rect[1] + y, win_rect[0] + x + width, win_rect[1] + y + height)
    screenshot = ImageGrab.grab(bbox=bbox)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def preprocess_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary_image

def process_image_with_ocr(image):
    processed_image = preprocess_image_for_ocr(image)
    # Use OCR with a whitelist of characters that match Minecraft coordinates
    return pytesseract.image_to_string(processed_image, config='--psm 7 -c tessedit_char_whitelist="0123456789.-/"')

def validate_and_correct_coordinate(value, coordinate_name):
    min_val, max_val = expected_ranges[coordinate_name]
    if min_val <= value <= max_val:
        return value
    print(f"Value {value} out of range for {coordinate_name}.")
    return None

def extract_coordinates_from_text(text):
    """Extract x, y, z coordinates from text using a consistent format."""
    # Find numbers in the format 'number / number / number'
    match = re.search(r"([-]?\d+\.\d+)\s*/\s*([-]?\d+\.\d+)\s*/\s*([-]?\d+\.\d+)", text)
    if match:
        x = validate_and_correct_coordinate(float(match.group(1)), 'x')
        y = validate_and_correct_coordinate(float(match.group(2)), 'y')
        z = validate_and_correct_coordinate(float(match.group(3)), 'z')
        if x is not None and y is not None and z is not None:
            return {'x': x, 'y': y, 'z': z}
    print("Coordinates not found or out of range in OCR text:", text)
    return None

def update_region_from_sliders(region):
    region['x_offset'] = cv2.getTrackbarPos("X Position", "Sliders")
    region['y_offset'] = cv2.getTrackbarPos("Y Position", "Sliders")
    region['width'] = cv2.getTrackbarPos("Width", "Sliders")
    region['height'] = cv2.getTrackbarPos("Height", "Sliders")
    save_region_config(region)

def create_slider_window(region):
    cv2.namedWindow("Sliders", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("X Position", "Sliders", region['x_offset'], 1920, lambda x: None)
    cv2.createTrackbar("Y Position", "Sliders", region['y_offset'], 1080, lambda x: None)
    cv2.createTrackbar("Width", "Sliders", region['width'], 1920, lambda x: None)
    cv2.createTrackbar("Height", "Sliders", region['height'], 1080, lambda x: None)

    while not stop_event.is_set():
        update_region_from_sliders(region)
        if cv2.waitKey(50) & 0xFF == ord('.'):
            stop_event.set()
            break

    cv2.destroyWindow("Sliders")

def display_preview_window(region, hwnd):
    cv2.namedWindow('Region Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Region Preview', 600, 400)
    try:
        while not stop_event.is_set():
            screenshot = capture_region(region, hwnd)
            cv2.imshow('Region Preview', screenshot)
            if cv2.waitKey(1) & 0xFF == ord('.'):
                stop_event.set()
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow('Region Preview')

def wait_for_start_key(start_key='s'):
    print(f"Press '{start_key.upper()}' to start OCR processing...")
    def on_press(key):
        try:
            if key.char == start_key:
                return False
        except AttributeError:
            pass
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    region = load_region_config()
    print(f"Loaded region configuration: {region}")

    hwnd = find_minecraft_window()
    if not hwnd:
        print("Minecraft game window not found. Please ensure the game is running.")
        return
    print("Found Minecraft game window.")

    slider_thread = Thread(target=create_slider_window, args=(region,))
    preview_thread = Thread(target=display_preview_window, args=(region, hwnd))
    slider_thread.start()
    preview_thread.start()

    wait_for_start_key('s')
    print("Starting OCR processing...")

    last_ocr_time = time.time()
    try:
        while not stop_event.is_set():
            screenshot = capture_region(region, hwnd)
            if time.time() - last_ocr_time >= 1:
                text = process_image_with_ocr(screenshot)
                coordinates = extract_coordinates_from_text(text)
                if coordinates:
                    print("Captured Coordinates:", coordinates)
                last_ocr_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('.'):
                stop_event.set()
                break
    except KeyboardInterrupt:
        print("Process terminated by user.")
    finally:
        stop_event.set()
        slider_thread.join()
        preview_thread.join()
        cv2.destroyAllWindows()
        print("Program exited cleanly.")

if __name__ == "__main__":
    main()

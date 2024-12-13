# shared_screenshot_manager.py
import threading
import mss
import numpy as np
import time
import multiprocessing

class SharedScreenshotManager:
    def __init__(self, bounds_list, capture_interval=0.05):
        self.bounds_list = bounds_list
        self.capture_interval = capture_interval
        self.full_bounds = self.compute_full_bounds()
        self.screenshot = None
        # Use multiprocessing Lock instead of threading Lock
        self.lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        self.thread = None  # Initialize thread in start()
        
        print("Full capture bounds:", self.full_bounds)

    def compute_full_bounds(self):
        left = min(b['left'] for b in self.bounds_list)
        top = min(b['top'] for b in self.bounds_list)
        right = max(b['left'] + b['width'] for b in self.bounds_list)
        bottom = max(b['top'] + b['height'] for b in self.bounds_list)
        width = right - left
        height = bottom - top
        return {'left': left, 'top': top, 'width': width, 'height': height}

    def start(self):
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def capture_loop(self):
        try:
            with mss.mss() as sct:
                while not self.stop_event.is_set():
                    try:
                        screenshot = np.array(sct.grab(self.full_bounds))[:, :, :3]
                        with self.lock:
                            self.screenshot = screenshot
                    except Exception as e:
                        print(f"Error capturing screenshot: {e}")
                        time.sleep(0.1)
                        continue
                    time.sleep(self.capture_interval)
        except Exception as e:
            print(f"Error in capture loop: {e}")

    def get_window_screenshot(self, window_bounds):
        with self.lock:
            if self.screenshot is None:
                return None
            try:
                x_offset = window_bounds['left'] - self.full_bounds['left']
                y_offset = window_bounds['top'] - self.full_bounds['top']
                width = window_bounds['width']
                height = window_bounds['height']
                window_image = self.screenshot[y_offset:y_offset+height, 
                                            x_offset:x_offset+width].copy()
                return window_image
            except Exception as e:
                print(f"Error extracting window image: {e}")
                return None
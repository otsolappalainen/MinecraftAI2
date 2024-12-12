# shared_screenshot_manager.py
import threading
import mss
import numpy as np
import time

class SharedScreenshotManager:
    def __init__(self, bounds_list, capture_interval=0.05):
        self.bounds_list = bounds_list  # List of window bounds
        self.capture_interval = capture_interval
        self.full_bounds = self.compute_full_bounds()
        self.screenshot = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.sct = mss.mss()

    def compute_full_bounds(self):
        # Calculate the bounding rectangle that encompasses all windows
        left = min(b['left'] for b in self.bounds_list)
        top = min(b['top'] for b in self.bounds_list)
        right = max(b['left'] + b['width'] for b in self.bounds_list)
        bottom = max(b['top'] + b['height'] for b in self.bounds_list)
        width = right - left
        height = bottom - top
        return {'left': left, 'top': top, 'width': width, 'height': height}

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sct.close()

    def capture_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                self.screenshot = np.array(self.sct.grab(self.full_bounds))[:, :, :3]
            time.sleep(self.capture_interval)

    def get_window_screenshot(self, window_bounds):
        with self.lock:
            if self.screenshot is None:
                return None
            x_offset = window_bounds['left'] - self.full_bounds['left']
            y_offset = window_bounds['top'] - self.full_bounds['top']
            width = window_bounds['width']
            height = window_bounds['height']
            window_image = self.screenshot[y_offset:y_offset+height, x_offset:x_offset+width]
            return window_image
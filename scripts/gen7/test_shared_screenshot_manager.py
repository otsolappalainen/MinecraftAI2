# test_shared_screenshot_manager.py

import threading
import mss
import numpy as np
import time
import pygetwindow as gw
import fnmatch
import cv2
import os
import psutil
from datetime import datetime
from collections import deque

# Import the SharedScreenshotManager class
from shared_screenshot_manager import SharedScreenshotManager

def find_minecraft_windows():
    print("=== Starting Minecraft Window Detection ===")
    patterns = ["Minecraft*", "*1.21.3*", "*Singleplayer*"]
    windows = []
    seen_handles = set()

    crop_size = 240  # Use the same crop size as in your code

    # Get all windows
    for title in gw.getAllTitles():
        for pattern in patterns:
            if fnmatch.fnmatch(title, pattern):
                for window in gw.getWindowsWithTitle(title):
                    if window._hWnd not in seen_handles:
                        center_x = window.left + window.width // 2
                        center_y = window.top + window.height // 2
                        half = crop_size // 2

                        window_info = {
                            "left": center_x - half,
                            "top": center_y - half,
                            "width": crop_size,
                            "height": crop_size,
                        }
                        windows.append(window_info)
                        seen_handles.add(window._hWnd)

    if len(windows) == 0:
        print("No Minecraft windows found.")
        return []

    print(f"Found {len(windows)} Minecraft windows.")
    return windows

def main():
    windows = find_minecraft_windows()
    if not windows:
        return

    print("\nWindow positions:")
    for idx, window in enumerate(windows):
        print(f"Window {idx}: ({window['left']}, {window['top']}) "
              f"{window['width']}x{window['height']}")

    screenshot_manager = SharedScreenshotManager(bounds_list=windows)
    screenshot_manager.start()

    time.sleep(0.5)

    try:
        output_dir = "window_screenshots"
        os.makedirs(output_dir, exist_ok=True)

        for idx, window_bounds in enumerate(windows):
            image = screenshot_manager.get_window_screenshot(window_bounds)
            if image is not None:
                # Image is already in BGR format for OpenCV
                image_path = os.path.join(output_dir, f"window_{idx}.png")
                cv2.imwrite(image_path, image)  # No need to flip channels
                print(f"Saved {image_path}")
            else:
                print(f"No image captured for window {idx}")
    finally:
        screenshot_manager.stop()

def stress_test(duration_seconds=10.0, capture_intervals=[0.05, 0.02, 0.01, 0.005]):
    windows = find_minecraft_windows()
    if not windows:
        return

    process = psutil.Process()
    results = []

    for interval in capture_intervals:
        print(f"\nTesting capture interval: {interval:.3f}s")
        
        # Initialize metrics
        fps_buffer = deque(maxlen=100)
        latency_buffer = deque(maxlen=100)
        cpu_usage = deque(maxlen=100)
        memory_usage = deque(maxlen=100)

        screenshot_manager = SharedScreenshotManager(bounds_list=windows, 
                                                  capture_interval=interval)
        screenshot_manager.start()

        start_time = time.time()
        frame_count = 0
        last_capture_time = None

        try:
            while (time.time() - start_time) < duration_seconds:
                loop_start = time.time()

                # Capture images from all windows
                for window_bounds in windows:
                    capture_start = time.time()
                    image = screenshot_manager.get_window_screenshot(window_bounds)
                    if image is not None:
                        capture_time = time.time() - capture_start
                        latency_buffer.append(capture_time * 1000)  # Convert to ms
                        frame_count += 1

                # Calculate FPS over last second
                elapsed = time.time() - loop_start
                if elapsed > 0:
                    fps_buffer.append(1.0 / elapsed)

                # Monitor system resources
                cpu_usage.append(process.cpu_percent())
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

                # Progress indicator
                if frame_count % 100 == 0:
                    current_fps = np.mean(fps_buffer) if fps_buffer else 0
                    current_latency = np.mean(latency_buffer) if latency_buffer else 0
                    print(f"\rFPS: {current_fps:.1f} | Latency: {current_latency:.1f}ms | "
                          f"CPU: {np.mean(cpu_usage):.1f}% | "
                          f"Memory: {np.mean(memory_usage):.1f}MB", end="")

        finally:
            screenshot_manager.stop()

        # Compute statistics
        results.append({
            'interval': interval,
            'fps_mean': np.mean(fps_buffer),
            'fps_std': np.std(fps_buffer),
            'latency_mean': np.mean(latency_buffer),
            'latency_std': np.std(latency_buffer),
            'cpu_mean': np.mean(cpu_usage),
            'cpu_std': np.std(cpu_usage),
            'memory_mean': np.mean(memory_usage),
            'memory_std': np.std(memory_usage)
        })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"stress_test_results_{timestamp}.txt", "w") as f:
        f.write("Stress Test Results\n")
        f.write("==================\n\n")
        for result in results:
            f.write(f"\nCapture Interval: {result['interval']:.3f}s\n")
            f.write(f"FPS: {result['fps_mean']:.1f} ± {result['fps_std']:.1f}\n")
            f.write(f"Latency: {result['latency_mean']:.1f} ± {result['latency_std']:.1f}ms\n")
            f.write(f"CPU Usage: {result['cpu_mean']:.1f} ± {result['cpu_std']:.1f}%\n")
            f.write(f"Memory Usage: {result['memory_mean']:.1f} ± {result['memory_std']:.1f}MB\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    # Test different capture intervals
    intervals = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005]  # 20, 50, 100, 200 FPS targets
    stress_test(duration_seconds=5.0, capture_intervals=intervals)
    main()
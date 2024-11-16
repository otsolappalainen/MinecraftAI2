import cv2
import numpy as np
from PIL import ImageGrab
import tkinter as tk
from tkinter import Scale

# Initialize Tkinter root for sliders
root = tk.Tk()
root.title("Adjust Box Coordinates")

# Set initial values
x_offset = tk.IntVar(value=66)
y_offset = tk.IntVar(value=308)
box_width = tk.IntVar(value=530)
box_height = tk.IntVar(value=29)

# Create sliders for x_offset, y_offset, width, height
tk.Label(root, text="X Offset").pack()
x_slider = Scale(root, from_=0, to=1200, variable=x_offset, orient="horizontal")
x_slider.pack()

tk.Label(root, text="Y Offset").pack()
y_slider = Scale(root, from_=0, to=900, variable=y_offset, orient="horizontal")
y_slider.pack()

tk.Label(root, text="Width").pack()
width_slider = Scale(root, from_=0, to=1200, variable=box_width, orient="horizontal")
width_slider.pack()

tk.Label(root, text="Height").pack()
height_slider = Scale(root, from_=0, to=900, variable=box_height, orient="horizontal")
height_slider.pack()

# Define screen capture function and display with rectangle overlay
def update_display():
    # Capture the screen region (1200x900 box from top left)
    screen_region = (0, 0, 1200, 900)
    screenshot = ImageGrab.grab(bbox=screen_region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get slider values for box coordinates
    x = x_offset.get()
    y = y_offset.get()
    w = box_width.get()
    h = box_height.get()

    # Draw a green rectangle on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Screen Region with Adjustable Box", frame)

    # Call update_display again after 10 ms
    root.after(10, update_display)

# Initialize display update
update_display()

# Start Tkinter event loop and display the window
root.mainloop()

# Release resources when the window is closed
cv2.destroyAllWindows()

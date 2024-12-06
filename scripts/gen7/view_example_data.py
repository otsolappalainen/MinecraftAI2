import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

EXAMPLE_DATA_DIR = "example_step_data"
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120


class DataViewer:
    def __init__(self, master, run_path):
        self.master = master
        self.master.title("Example Step Data Viewer")
        self.run_path = run_path

        # Gather all samples
        self.samples = sorted([d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))])
        self.total_samples = len(self.samples)
        self.current_index = 0

        # Set up UI
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.text_label = tk.Label(master, justify=tk.LEFT, font=("Courier", 10))
        self.text_label.pack()

        # Bind keys
        self.master.bind('<i>', self.next_sample)
        self.master.bind('<o>', self.prev_sample)

        if self.total_samples > 0:
            self.display_sample(self.current_index)
        else:
            self.text_label.config(text="No samples available.")

    def display_sample(self, index):
        sample_dir = os.path.join(self.run_path, self.samples[index])

        # Load and display images
        step_images = []
        step_texts = []
        for step_num in sorted(os.listdir(sample_dir)):
            if step_num.endswith("_image.png"):
                img_path = os.path.join(sample_dir, step_num)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = image.resize((IMAGE_WIDTH*5, IMAGE_HEIGHT*5), Image.NEAREST)
                tk_image = ImageTk.PhotoImage(image)
                step_images.append(tk_image)

            elif step_num.endswith("_data.json"):
                json_path = os.path.join(sample_dir, step_num)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                step_texts.append(json.dumps(data, indent=4))

        # Display the first step's image and data
        if step_images and step_texts:
            self.current_sample_images = step_images
            self.current_sample_texts = step_texts
            self.current_step = 0
            self.update_display()
        else:
            self.text_label.config(text="Incomplete sample data.")

    def update_display(self):
        # Update image
        img = self.current_sample_images[self.current_step]
        self.image_label.config(image=img)
        self.image_label.image = img  # Keep a reference

        # Update text
        text = self.current_sample_texts[self.current_step]
        self.text_label.config(text=text)

    def next_sample(self, event=None):
        if self.current_index < self.total_samples - 1:
            self.current_index += 1
            self.display_sample(self.current_index)

    def prev_sample(self, event=None):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_sample(self.current_index)

def select_run_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    run_dir = filedialog.askdirectory(title="Select Run Directory", initialdir=EXAMPLE_DATA_DIR)
    root.destroy()
    return run_dir

if __name__ == "__main__":
    run_path = select_run_directory()
    if not run_path:
        print("No directory selected.")
        exit()

    root = tk.Tk()
    viewer = DataViewer(root, run_path)
    root.mainloop()
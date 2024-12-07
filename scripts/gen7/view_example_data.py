import os
import json
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

EXAMPLE_DATA_DIR = "example_step_data"
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120


class DataViewer:
    def __init__(self, master):
        self.master = master
        self.master.title("Example Step Data Viewer")
        self.run_path = EXAMPLE_DATA_DIR

        # Gather all steps from all runs and samples
        self.steps = self.gather_all_steps(self.run_path)
        self.total_steps = len(self.steps)
        self.current_index = 0

        # Set up UI
        self.image_label = tk.Label(master)
        self.image_label.pack()

        self.text_label = tk.Label(master, justify=tk.LEFT, font=("Courier", 10))
        self.text_label.pack()

        self.info_label = tk.Label(master, justify=tk.LEFT, font=("Courier", 10))
        self.info_label.pack()

        # Bind keys
        self.master.bind('<i>', self.next_step)  # Next step
        self.master.bind('<o>', self.prev_step)  # Previous step

        if self.total_steps > 0:
            self.display_step(self.current_index)
        else:
            self.text_label.config(text="No steps available.")

    def gather_all_steps(self, run_path):
        """Gather all steps from all runs and samples in the run_path directory."""
        steps = []
        if not os.path.exists(run_path):
            print(f"Run path '{run_path}' does not exist.")
            return steps

        run_dirs = sorted([d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))])
        if not run_dirs:
            print(f"No run directories found in '{run_path}'.")
            return steps

        for run in run_dirs:
            run_dir = os.path.join(run_path, run)
            sample_dirs = sorted([d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))])
            if not sample_dirs:
                print(f"No sample directories found in run '{run}'. Skipping.")
                continue

            for sample in sample_dirs:
                sample_dir = os.path.join(run_dir, sample)
                step_files = sorted([f for f in os.listdir(sample_dir) if f.endswith("_image.png")])

                if not step_files:
                    print(f"No step image files found in sample '{sample}' of run '{run}'. Skipping.")
                    continue

                for file in step_files:
                    # Correct step_num extraction
                    parts = file.split("_")
                    if len(parts) < 3:
                        print(f"Unexpected file format '{file}' in sample '{sample}' of run '{run}'. Skipping step.")
                        continue

                    step_num = parts[1]
                    image_path = os.path.join(sample_dir, file)
                    json_filename = f"step_{step_num}_data.json"
                    json_path = os.path.join(sample_dir, json_filename)

                    if not os.path.exists(json_path):
                        print(f"Missing JSON file for step '{step_num}' in sample '{sample}' of run '{run}'. Skipping step.")
                        continue

                    steps.append({
                        "run": run,
                        "sample": sample,
                        "step_num": step_num,
                        "image_path": image_path,
                        "json_path": json_path
                    })

        return steps

    def display_step(self, index):
        """Display the step at the given index."""
        if index < 0 or index >= self.total_steps:
            print(f"Index {index} is out of bounds.")
            return

        step = self.steps[index]
        run = step["run"]
        sample = step["sample"]
        step_num = step["step_num"]

        # Load and display image
        img_path = step["image_path"]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image at '{img_path}'.")
            self.text_label.config(text=f"Failed to load image at '{img_path}'.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((IMAGE_WIDTH * 5, IMAGE_HEIGHT * 5), Image.NEAREST)
        tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image  # Keep a reference

        # Load and display JSON data
        json_path = step["json_path"]
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            formatted_json = json.dumps(data, indent=4)
            self.text_label.config(text=formatted_json)
        except Exception as e:
            print(f"Error loading JSON file '{json_path}': {e}")
            self.text_label.config(text=f"Error loading JSON data: {e}")

        # Update info label
        self.info_label.config(text=f"Run: {run} | Sample: {sample} | Step: {step_num} ({index + 1}/{self.total_steps})")

    def next_step(self, event=None):
        """Navigate to the next step."""
        if self.current_index < self.total_steps - 1:
            self.current_index += 1
            self.display_step(self.current_index)
        else:
            print("Already at the last step.")

    def prev_step(self, event=None):
        """Navigate to the previous step."""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_step(self.current_index)
        else:
            print("Already at the first step.")


def main():
    if not os.path.exists(EXAMPLE_DATA_DIR):
        print(f"Directory '{EXAMPLE_DATA_DIR}' does not exist.")
        return

    root = tk.Tk()
    viewer = DataViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
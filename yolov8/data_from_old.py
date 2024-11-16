import os
import shutil

# Define the old and new paths
old_paths = {
    "images_train": r"C:\Users\odezz\source\MinecraftAI\yolov5\data\images\train",
    "images_val": r"C:\Users\odezz\source\MinecraftAI\yolov5\data\images\val",
    "labels_train": r"C:\Users\odezz\source\MinecraftAI\yolov5\data\labels\train",
    "labels_val": r"C:\Users\odezz\source\MinecraftAI\yolov5\data\labels\val"
}

new_paths = {
    "images_train": r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\images\train",
    "images_val": r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\images\val",
    "labels_train": r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\labels\train",
    "labels_val": r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\labels\val"
}

# Ensure new directories exist
for path in new_paths.values():
    os.makedirs(path, exist_ok=True)

# Function to move files from old path to new path
def move_files(old_dir, new_dir):
    for filename in os.listdir(old_dir):
        src_path = os.path.join(old_dir, filename)
        dest_path = os.path.join(new_dir, filename)
        if os.path.isfile(src_path):
            shutil.move(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")

# Move the files
move_files(old_paths["images_train"], new_paths["images_train"])
move_files(old_paths["images_val"], new_paths["images_val"])
move_files(old_paths["labels_train"], new_paths["labels_train"])
move_files(old_paths["labels_val"], new_paths["labels_val"])

print("All files moved successfully.")

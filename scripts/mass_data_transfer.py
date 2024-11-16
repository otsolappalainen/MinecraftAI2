import os
import shutil
import random
from collections import defaultdict

# Define paths
base_dir = "C:/Users/odezz/source/MinecraftAI2/yolov8/data"
image_dirs = {"train": os.path.join(base_dir, "images/train"),
              "val": os.path.join(base_dir, "images/val")}
label_dirs = {"train": os.path.join(base_dir, "labels/train"),
              "val": os.path.join(base_dir, "labels/val")}

# New source directory
source_dir = "C:/Users/odezz/source/MinecraftAI2/scripts/captured_data"

excess_data_dir = os.path.join(base_dir, "excess_data")
os.makedirs(excess_data_dir, exist_ok=True)

# Function to move existing data to excess_data folder
def move_existing_data():
    for dir_path in image_dirs.values():
        if os.listdir(dir_path):  # Check if folder is not empty
            excess_path = os.path.join(excess_data_dir, os.path.basename(dir_path))
            os.makedirs(excess_path, exist_ok=True)
            for file_name in os.listdir(dir_path):
                shutil.move(os.path.join(dir_path, file_name), excess_path)
    for dir_path in label_dirs.values():
        if os.listdir(dir_path):  # Check if folder is not empty
            excess_path = os.path.join(excess_data_dir, os.path.basename(dir_path))
            os.makedirs(excess_path, exist_ok=True)
            for file_name in os.listdir(dir_path):
                shutil.move(os.path.join(dir_path, file_name), excess_path)

# Prompt the user if existing data is found
def check_existing_data():
    data_exists = any(os.listdir(dir_path) for dir_path in image_dirs.values()) or \
                  any(os.listdir(dir_path) for dir_path in label_dirs.values())
    if data_exists:
        choice = input("Existing data found in target folders. Do you want to (m)ove it to excess data, (d)elete it, (k)eep it, or (c)ancel? ").lower()
        if choice == 'm':
            print("Moving existing data to excess_data folder...")
            move_existing_data()
        elif choice == 'd':
            print("Deleting existing data...")
            for dir_path in image_dirs.values():
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
            for dir_path in label_dirs.values():
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        elif choice == 'k':
            print("Keeping existing data. New data will be added alongside the old data.")
        elif choice == 'c':
            print("Operation canceled.")
            exit()
        else:
            print("Invalid choice. Operation canceled.")
            exit()

# Create target directories if they don't exist
for dir_path in image_dirs.values():
    os.makedirs(dir_path, exist_ok=True)
for dir_path in label_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Check for existing data and prompt the user
check_existing_data()

# List of images and labels
image_files = sorted([f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
label_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.txt')])

# Match images and labels by name (without extensions)
paired_files = [(img, img.replace(img.split('.')[-1], 'txt')) for img in image_files if img.replace(img.split('.')[-1], 'txt') in label_files]

# Shuffle and split data into training and validation sets (80-20 split)
random.shuffle(paired_files)
split_idx = int(0.8 * len(paired_files))
train_files = paired_files[:split_idx]
val_files = paired_files[split_idx:]

# Initialize class counts
class_counts = {"train": defaultdict(int), "val": defaultdict(int)}

# Function to count classes in label files
def count_classes(label_path, dataset_type):
    with open(label_path, 'r') as f:
        for line in f:
            class_id = line.split()[0]
            class_counts[dataset_type][class_id] += 1

# Move files to respective folders
def move_files(file_pairs, dataset_type):
    for img_file, lbl_file in file_pairs:
        img_src = os.path.join(source_dir, img_file)
        lbl_src = os.path.join(source_dir, lbl_file)
        
        img_dest = os.path.join(image_dirs[dataset_type], img_file)
        lbl_dest = os.path.join(label_dirs[dataset_type], lbl_file)
        
        shutil.move(img_src, img_dest)
        shutil.move(lbl_src, lbl_dest)
        
        # Count classes
        count_classes(lbl_dest, dataset_type)

# Move training and validation files
move_files(train_files, "train")
move_files(val_files, "val")

# Display class counts
print("Class counts in train set:")
for cls, count in class_counts["train"].items():
    print(f"Class {cls}: {count} images")

print("\nClass counts in val set:")
for cls, count in class_counts["val"].items():
    print(f"Class {cls}: {count} images")
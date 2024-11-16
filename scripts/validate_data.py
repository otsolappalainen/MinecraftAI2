import os
import glob
from collections import Counter

# Define the paths
data_dir = "C:/Users/odezz/source/MinecraftAI2/yolov8/data"
image_train_dir = os.path.join(data_dir, "images/train")
image_val_dir = os.path.join(data_dir, "images/val")
label_train_dir = os.path.join(data_dir, "labels/train")
label_val_dir = os.path.join(data_dir, "labels/val")

class_names = {
    0: "valuable block",
    1: "torch",
    2: "lava",
    3: "water"
}

color_codes = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
reset_code = "\033[0m"


# Function to ensure each image has a corresponding label and vice versa
def check_correspondence(image_dir, label_dir):
    images = set(os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg"))
    labels = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt"))

    missing_images = labels - images
    missing_labels = images - labels

    return missing_images, missing_labels

# Function to check classes in label files
def check_and_relabel_classes(label_dir, valid_classes=range(5)):
    class_counter = Counter()
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(label_file, "r") as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id in valid_classes:
                new_lines.append(line)
                class_counter[class_id] += 1
            else:
                print(f"Invalid class {class_id} found in {label_file}, skipping line.")

        # Overwrite the file with valid classes only
        with open(label_file, "w") as file:
            file.writelines(new_lines)

    return class_counter

# Check train and val directories
train_missing_images, train_missing_labels = check_correspondence(image_train_dir, label_train_dir)
val_missing_images, val_missing_labels = check_correspondence(image_val_dir, label_val_dir)

# Count classes in train and val
train_class_count = check_and_relabel_classes(label_train_dir, valid_classes=range(4))
val_class_count = check_and_relabel_classes(label_val_dir, valid_classes=range(4))

def print_class_distribution(class_counts, dataset_type):
    print(f"\n{dataset_type.capitalize()} Class Distribution:")
    print("-" * 30)
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, "unknown")
        color = color_codes[class_id % len(color_codes)]  # Choose color for each class
        print(f"Class {class_id} ({class_name:<15}) {color}{count:>10}{reset_code}")


print_class_distribution(train_class_count, "train")
print_class_distribution(val_class_count, "val")

print("-" * 30)
print(f"Train - Missing images for labels: {len(train_missing_images)}")
print(f"Train - Missing labels for images: {len(train_missing_labels)}")
print(f"Val - Missing images for labels: {len(val_missing_images)}")
print(f"Val - Missing labels for images: {len(val_missing_labels)}")
print("-" * 30)

# Ask user if they want to delete unmatched files with confirmation
def delete_files(file_list, file_type):
    for file in file_list:
        full_path = f"{file}.{file_type}"
        if os.path.exists(full_path):
            response = input(f"Do you want to delete {full_path}? (y/n): ").strip().lower()
            if response == 'y':
                os.remove(full_path)
                print(f"Deleted: {full_path}")
            else:
                print(f"Skipped: {full_path}")

if train_missing_images or train_missing_labels or val_missing_images or val_missing_labels:
    response = input("Do you want to delete labels with no corresponding image files and images with no corresponding label files? (y/n): ").strip().lower()
    if response == 'y':
        delete_files([os.path.join(label_train_dir, file) for file in train_missing_images], "txt")
        delete_files([os.path.join(image_train_dir, file) for file in train_missing_labels], "png")
        delete_files([os.path.join(label_val_dir, file) for file in val_missing_images], "txt")
        delete_files([os.path.join(image_val_dir, file) for file in val_missing_labels], "png")
    else:
        print("No files were deleted.")

print("Cleanup complete.")

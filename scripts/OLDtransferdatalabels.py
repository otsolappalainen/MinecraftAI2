import os
from collections import Counter

# Paths to your image and label directories
base_dir = "C:/Users/odezz/source/MinecraftAI2/yolov8/data"
image_dirs = {"train": os.path.join(base_dir, "images/train"),
              "val": os.path.join(base_dir, "images/val")}
label_dirs = {"train": os.path.join(base_dir, "labels/train"),
              "val": os.path.join(base_dir, "labels/val")}

# Mapping of old classes to new classes
class_mapping = {
    "0": "0",  # "valuable block" -> (diamonds, gold, redstone)
    "1": "2",  # "torch" stays as "torch"
    "2": "1",  # "lava" stays as "lava"
    "3": "0",  # Assuming gold and redstone as valuable block
    "4": "0"   # Assuming redstone as valuable block
}

# Initialize counters
class_counts = {"train": Counter(), "val": Counter()}

# Process images and labels
for split in ["train", "val"]:
    image_dir = image_dirs[split]
    label_dir = label_dirs[split]

    # Ensure each image has a corresponding label file
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            label_file = image_file.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            # Check if label file exists
            if os.path.exists(label_path):
                # Process label file
                with open(label_path, "r") as file:
                    lines = file.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts and parts[0] in class_mapping:
                        new_class = class_mapping[parts[0]]
                        new_lines.append(f"{new_class} " + " ".join(parts[1:]))
                        class_counts[split][new_class] += 1
                
                # Rewrite the label file with updated classes
                with open(label_path, "w") as file:
                    file.write("\n".join(new_lines))
            else:
                print(f"Warning: Missing label file for image {image_file}")

# Display class counts
print("Class counts in train set:")
for cls, count in class_counts["train"].items():
    print(f"Class {cls}: {count} images")

print("\nClass counts in val set:")
for cls, count in class_counts["val"].items():
    print(f"Class {cls}: {count} images")

import os
import cv2
import numpy as np
from pathlib import Path

# Base directory for the new project structure
base_dir = Path(r"C:/Users/odezz/source/MinecraftAI2/yolov8")

def list_subdirectories(base_path):
    """List subdirectories in a given path and return them as a list."""
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    for idx, subdir in enumerate(subdirs):
        print(f"{idx + 1}: {subdir.name}")
    return subdirs

def select_directory(base_path, prompt):
    """Prompt the user to select a directory."""
    print(prompt)
    subdirs = list_subdirectories(base_path)
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: ")) - 1
            if 0 <= choice < len(subdirs):
                return subdirs[choice]
            else:
                print("Invalid choice, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")

def overlay_bounding_boxes(image_path, label_path, margin_ratio=0.3):
    """Overlay bounding boxes on the image using YOLO labels with refined margin handling."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_h, img_w = image.shape[:2]
    
    # Read the label file
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid label format in {label_path}")
                continue
            
            class_id, x_center, y_center, width, height = parts
            x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
            
            # Convert YOLO format to bounding box coordinates
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)
            
            # Add margin to the bounding box for clearer context
            width_margin = int((x2 - x1) * margin_ratio)
            height_margin = int((y2 - y1) * margin_ratio)
            x1 = max(0, x1 - width_margin)
            y1 = max(0, y1 - height_margin)
            x2 = min(img_w, x2 + width_margin)
            y2 = min(img_h, y2 + height_margin)
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Class: {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    return image

def process_directory(images_dir, labels_dir):
    overlay_dir = images_dir / "overlayed"
    overlay_dir.mkdir(exist_ok=True)

    for filename in os.listdir(images_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = images_dir / filename
            label_path = labels_dir / filename.replace(".png", ".txt").replace(".jpg", ".txt")
            
            if not label_path.exists():
                print(f"No label file found for {filename}")
                continue
            
            image_with_boxes = overlay_bounding_boxes(image_path, label_path)
            if image_with_boxes is not None:
                cv2.imshow("Bounding Box Overlay", image_with_boxes)
                
                overlay_image_path = overlay_dir / filename
                cv2.imwrite(str(overlay_image_path), image_with_boxes)
                print(f"Overlay saved: {overlay_image_path}")

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

def main():
    print("Select the images directory:")
    images_dir = select_directory(base_dir / "data/images", "Select the 'train' or 'val' directory under 'images'.")
    
    print("\nSelect the labels directory:")
    labels_dir = select_directory(base_dir / "data/labels", "Select the 'train' or 'val' directory under 'labels'.")
    
    print("Processing images in:", images_dir)
    print("Using labels from:", labels_dir)
    process_directory(images_dir, labels_dir)

if __name__ == "__main__":
    main()

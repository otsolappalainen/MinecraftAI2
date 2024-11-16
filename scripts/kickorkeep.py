import os
import cv2

# Directories for images and labels
train_directory = r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\images\train"
val_directory = r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\images\val"
label_train_directory = r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\labels\train"
label_val_directory = r"C:\Users\odezz\source\MinecraftAI2\yolov8\data\labels\val"

# Class mapping: replace class IDs with names
class_mapping = {
    0: 'valuable block',
    1: 'torch',
    2: 'lava',
    3: 'water'
}

def get_image_label_paths(image_path):
    """Get the corresponding label file path for a given image file."""
    if "train" in image_path:
        label_path = os.path.join(label_train_directory, os.path.basename(image_path).replace(".png", ".txt").replace(".jpg", ".txt"))
    else:
        label_path = os.path.join(label_val_directory, os.path.basename(image_path).replace(".png", ".txt").replace(".jpg", ".txt"))
    return label_path

def draw_boxes(image, label_path):
    """Draw bounding boxes from YOLO label file on the image."""
    img_h, img_w = image.shape[:2]
    if not os.path.exists(label_path):
        print(f"Label file not found for image {label_path}")
        return image
    
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)
            
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Get the class name and add to the label
            class_name = class_mapping.get(int(class_id), "Unknown")
            label_text = f"{class_name}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

def review_images(directory):
    """Review images in a directory, with options to delete or keep."""
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            label_path = get_image_label_paths(image_path)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image {image_path}")
                continue

            # Draw bounding boxes from label file
            image_with_boxes = draw_boxes(image.copy(), label_path)

            # Show the image with boxes
            cv2.imshow("Image Review (Press 'd' to delete, 'k' to keep, 'q' to quit)", image_with_boxes)
            
            # Wait for key press
            key = cv2.waitKey(0)
            
            if key == ord('d'):  # Delete the image and label
                os.remove(image_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                print(f"Deleted {image_path} and {label_path}")
                
            elif key == ord('k'):  # Keep the image
                print(f"Kept {image_path}")

            elif key == ord('q'):  # Quit the loop
                break

    cv2.destroyAllWindows()

# Review images in the train and validation directories
print("Reviewing training images...")
review_images(train_directory)
print("Reviewing validation images...")
review_images(val_directory)

print("Image review completed.")

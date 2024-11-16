import os
import cv2

# Directory for screenshots and label file path
SCREENSHOT_DIR = "minecraft_screenshots"
LABEL_FILE = "labeled_coordinates.txt"

# Ensure the directory and label file exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def load_unlabeled_images():
    """Load image names that haven't been labeled yet."""
    # Load existing labels from the file
    labeled_images = set()
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r") as f:
            for line in f:
                image_name = line.split(",")[0].strip()
                labeled_images.add(image_name)
    
    # Filter out labeled images
    all_images = sorted(os.listdir(SCREENSHOT_DIR))
    unlabeled_images = [img for img in all_images if img not in labeled_images]
    
    # Ensure that unlabeled images are listed in the label file in the right order
    with open(LABEL_FILE, "a") as f:
        for img in unlabeled_images:
            f.write(f"{img},\n")
    
    return unlabeled_images

def display_images(images):
    """Display each unlabeled image one by one and print its name in the console."""
    for image_name in images:
        image_path = os.path.join(SCREENSHOT_DIR, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load image {image_path}. Skipping.")
            continue

        # Print the current image name to the console
        print(f"Displaying {image_name}. Label it in {LABEL_FILE}")

        # Display the image
        cv2.imshow("Labeling Tool", image)
        cv2.waitKey(0)  # Wait until the window is closed

    print("All images have been displayed.")

    # Close any remaining OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load only unlabeled images
    unlabeled_images = load_unlabeled_images()
    
    if not unlabeled_images:
        print("No unlabeled images found. All images in the directory have been labeled.")
    else:
        # Display images one by one
        display_images(unlabeled_images)


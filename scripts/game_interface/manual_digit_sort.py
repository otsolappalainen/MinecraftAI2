import cv2
import os
import shutil

# Paths
input_dir = "split_digits"       # Folder with the split images
output_dir = "cropped_digits"    # Folder where labeled images will be saved

# Define character classes and create subdirectories
label_classes = "0123456789-/. ()"
for label in label_classes:
    label_folder = {
        " ": "space",
        ".": "dot",
        "/": "slash",
        "(": "left_paren",
        ")": "right_paren"
    }.get(label, label)  # Use the label itself as folder name if not special
    os.makedirs(os.path.join(output_dir, label_folder), exist_ok=True)

# Function to classify and save each image
def classify_and_save_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    # Show the image in a resizable window
    cv2.namedWindow("Classify Digit", cv2.WINDOW_NORMAL)
    cv2.imshow("Classify Digit", image)
    print("Enter the correct label for the displayed image (0-9, '-', '/', '.', '(', ')', or space for blank):")
    key = cv2.waitKey(0)

    # Determine the label based on user input
    if key == ord(' '):
        label = "space"
    elif key == ord('.'):
        label = "dot"
    elif key == ord('/'):
        label = "slash"
    elif key == ord('('):
        label = "left_paren"
    elif key == ord(')'):
        label = "right_paren"
    elif chr(key) in label_classes:
        label = chr(key)
    else:
        print("Invalid input. Skipping this image.")
        return

    # Move the image to the correct labeled folder
    label_folder = {
        "space": "space",
        "dot": "dot",
        "slash": "slash",
        "left_paren": "left_paren",
        "right_paren": "right_paren"
    }.get(label, label)
    save_path = os.path.join(output_dir, label_folder, os.path.basename(image_path))
    shutil.move(image_path, save_path)
    print(f"Image saved to {save_path}")

# Iterate over each image in the input directory
for filename in sorted(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, filename)
    classify_and_save_image(image_path)

# Cleanup
cv2.destroyAllWindows()
print("Image classification and sorting completed.")




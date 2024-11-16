import cv2
import numpy as np
import os

# Directory to save individual character images
output_dir = "split_digits"
os.makedirs(output_dir, exist_ok=True)

# Directory with the input images
input_dir = r"C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\minecraft_screenshots"  # Replace with the actual path to your screenshots folder

# Process each image in the input directory
for filename in sorted(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, filename)

    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
        continue

    # Apply binary thresholding to make characters clear
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

    # Get the height and width of the image
    height, width = binary_image.shape

    # Set up variables for character segmentation
    character_segments = []
    start_col = None

    # Loop through each column to detect character segments based on black pixel counts
    for col in range(width):
        black_count = np.sum(binary_image[:, col] == 0)    # Count black pixels (characters)

        # Detect gaps of at least 10 pixels as spaces
        if black_count == 0:
            if start_col is not None:
                # End of character found, add segment
                character_segments.append((start_col, col - 1))
                start_col = None  # Reset for the next character
        else:
            if start_col is None:
                # Start a new character segment
                start_col = col

    # Save the last character segment if the image ends with it
    if start_col is not None:
        character_segments.append((start_col, width - 1))

    # Initialize the last_end to track the position of the last character's end column
    last_end = None

    # Save each character segment as an individual file
    for idx, (start, end) in enumerate(character_segments):
        # Check if there was a gap of at least 10 pixels before the current character segment
        if last_end is not None and start - last_end >= 10:
            # Save a "space" image for gaps 10 pixels or wider
            space_image = 255 * np.ones((height, 10), dtype=np.uint8)  # Create a small white image for space
            space_path = os.path.join(output_dir, f"{filename}_space_{idx}.png")
            cv2.imwrite(space_path, space_image)
            print(f"Saved space image: {space_path}")

        # Crop the character from the binary image
        character_image = binary_image[:, start:end+1]  # Include the last column

        # Generate a unique file name for each character image
        save_path = os.path.join(output_dir, f"{filename}_character_{idx}.png")
        cv2.imwrite(save_path, character_image)
        print(f"Saved character image: {save_path}")

        # Update last_end for the next iteration
        last_end = end

print("Character segmentation completed.")

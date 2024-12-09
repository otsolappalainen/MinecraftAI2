# preprocess_images.py
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120

def process_image(image_path, output_path):
    """Process a single image to match Minecraft environment format"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image: {image_path}")
            return False
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # Save
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for CNN visualization')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--output_dir', default='test_images', 
                        help='Output directory for processed images')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process all images
    input_dir = Path(args.input_dir)
    supported_formats = {'.png', '.jpg', '.jpeg'}
    
    processed = 0
    failed = 0
    
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in supported_formats:
            output_path = output_dir / img_path.name
            
            print(f"Processing: {img_path.name}")
            success = process_image(img_path, output_path)
            
            if success:
                processed += 1
            else:
                failed += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Failed: {failed} images")
    print(f"Processed images saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
import pytesseract
from PIL import Image

# Update this path to an image file on your system
image_path = r'C:\Users\odezz\source\MinecraftAI2\scripts\game_interface\image.png'

# Open the image
img = Image.open(image_path)

# Perform OCR
text = pytesseract.image_to_string(img)

print(text)
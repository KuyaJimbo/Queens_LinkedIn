import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

def show_grid(image):
  # Check if the image was loaded successfully
  if image is None:
      print("Error: Image not found.")
  else:
      # Display images using OpenCV's cv2_imshow
      print("Original Image:")
      # cv2_imshow(image)

      # Image Width and Height
      image_height, image_width, _ = image.shape
      print(f"Image Width: {image_width} pixels")
      print(f"Image Height: {image_height} pixels")

      cropped_image = image[150:518, 15:385]

      # Display the cropped image
      cv2_imshow(cropped_image)

# Load the images
image_paths = ["queen1.png","queen2.png"]

for path in image_paths:
  image = cv2.imread(path)
  show_grid(image)

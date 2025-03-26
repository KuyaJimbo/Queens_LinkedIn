import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

def get_grid_image(path):
  image = cv2.imread(path)
  # Check if the image was loaded successfully
  if image is None:
      print("Error: Image not found.")
      return False
  else:
      # Get image dimensions
      image_height, image_width, _ = image.shape
      print(f"Image Width: {image_width} pixels")
      print(f"Image Height: {image_height} pixels")

      # Define cropping percentages (based on the example values)
      top_pct = 150 / 866   # 17.3% from the top
      bottom_pct = 518 / 866  # 59.8% from the top
      left_pct = 15 / 400   # 3.75% from the left
      right_pct = 385 / 400  # 96.25% from the left

      # Convert percentages to pixel values dynamically
      top = int(image_height * top_pct)
      bottom = int(image_height * bottom_pct)
      left = int(image_width * left_pct)
      right = int(image_width * right_pct)

      # Perform the cropping
      cropped_image = image[top:bottom, left:right]

      # Display the cropped image
      print("Cropped Image (Percentage-based):")
      cv2_imshow(cropped_image)
      return cropped_image

grid_image = get_grid_image("queen1.png")
if grid_image.any():
  print("Good")
else:
  print("Bad")
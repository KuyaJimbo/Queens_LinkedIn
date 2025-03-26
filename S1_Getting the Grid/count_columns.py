import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
import cv2

def count_columns(image):
    # Convert the line to grayscale if it's a color image
    if len(image.shape) == 3:
        line = cv2.cvtColor(image[10:11, :], cv2.COLOR_BGR2GRAY)
    else:
        line = image[10:11, :]

    # Flatten the line to 1D array
    line_1d = line.flatten()
    
    # Threshold to determine black pixels (assuming black is close to 0)
    black_threshold = 100  # Adjust this value based on your image's characteristics
    
    # Identify black and non-black pixels
    is_black = line_1d <= black_threshold
    
    # Count columns by detecting transitions from black to non-black
    columns = 0
    in_border = True
    
    for pixel in is_black:
        if pixel and not in_border:
            # Transition from non-black to black (border start)
            in_border = True
        elif not pixel and in_border:
            # Transition from black to non-black (column start)
            columns += 1
            in_border = False
    

    cv2_imshow(line)
    print(f"Number of columns detected: {columns}")
    return columns

def crop_image(path):
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

        return cropped_image

# Main execution
grid_image = crop_image("queen1.png")
size = count_columns(grid_image)
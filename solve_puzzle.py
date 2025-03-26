import cv2
import numpy as np
from PIL import Image
import io
from extract_info import preprocess_image, crop_image, count_columns, color_map
from reduce_grid import reduce
from backtracking import solve

# Replace the google.colab.patches import with PIL for image display
def cv2_imshow(image):
    # Convert OpenCV image to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


# Display the solution
# Modify display_solution to use consistent scaling
def display_solution(color_solution, grid_image, w, h, size):
    # Case 1: No solution
    if color_solution is None:
        return None
    
    # Case 2: Solution Found
    # Create a copy of the original image
    solution_image = grid_image.copy()

    # Mark all queen positions on the image with a black circle
    for color, (row, col) in color_solution.items():
        # Calculate the center of the cell using consistent scaling
        x = int((col + 0.5) / size * w)
        y = int((row + 0.5) / size * h)
        
        # Draw a black circle at the center of the cell
        # Adjust circle size proportionally to image width
        radius = int(max(10, w * 0.025))  # 2.5% of image width, minimum 10 pixels
        cv2.circle(solution_image, (x, y), radius, (0, 0, 0), -1)

    return solution_image

# Modify solve_queens_puzzle to accommodate the new color_map return
def solve_queens_puzzle(image_array):
    try:
        # Preprocess the image to target dimensions
        preprocessed_image = preprocess_image(image_array)

        # Crop the image
        grid_image, w, h = crop_image(preprocessed_image)

        # Count the number of columns in the grid
        size = count_columns(grid_image)

        # Create a color grid based on the image
        color_grid, residesDict, color_rows, color_columns, sample_image = color_map(grid_image, size, w, h)

        # Reduce the grid based on the color constraints
        reduced_grid, residesDict, color_rows, color_columns = reduce(residesDict, color_rows, color_columns, size, color_grid)

        # Solve the puzzle
        color_solution = solve(reduced_grid, residesDict)

        # Display the solution
        solution_image = display_solution(color_solution, grid_image, w, h, size)

        return solution_image, sample_image
    
    except Exception as e:
        print(f"Error solving puzzle: {e}")
        return None, None


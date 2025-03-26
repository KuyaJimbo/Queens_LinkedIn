import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

def count_columns(image):
    # Convert the line to grayscale if it's a color image
    if len(image.shape) == 3:
        line = cv2.cvtColor(image[10:11, :], cv2.COLOR_BGR2GRAY)
    else:
        line = image[10:11, :]

    # Threshold the image to create a binary representation
    _, binary_line = cv2.threshold(line, 100, 255, cv2.THRESH_BINARY)
    
    # Flatten the binary line
    line_1d = binary_line.flatten()
    
    # Count columns by detecting transitions from black to white
    columns = 0
    in_border = True
    
    for pixel in line_1d:
        if pixel == 0 and not in_border:
            # Transition from white to black (border start)
            in_border = True
        elif pixel == 255 and in_border:
            # Transition from black to white (column start)
            columns += 1
            in_border = False
    
    # Visualize the line
    plt.figure(figsize=(10, 2))
    plt.imshow(binary_line, cmap='binary')
    plt.title(f'Analyzed Line (Columns: {columns})')
    plt.show()
    
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
        top_pct = 150 / image_height   # 17.3% from the top
        bottom_pct = 518 / image_height  # 59.8% from the top
        left_pct = 15 / image_width   # 3.75% from the left
        right_pct = 385 / image_width  # 96.25% from the left

        # Convert percentages to pixel values dynamically
        top = int(image_height * top_pct)
        bottom = int(image_height * bottom_pct)
        left = int(image_width * left_pct)
        right = int(image_width * right_pct)

        # final width and height
        width = right - left
        height = bottom - top

        # Perform the cropping
        cropped_image = image[top:bottom, left:right]

        return cropped_image, width, height

def color_map(image, size, w, h):
    # Create a 2D array of 0's with size x size
    color_grid = np.zeros((size, size), dtype=int)
    color_set = []      
    residesDict = dict()  # key = index of color_set, value = list of row and column pairs on the color_grid
    color_rows = dict()   # key = index of color_set, value = list of rows where the color is present on the color_grid
    color_columns = dict()   # key = index of color_set, value = list of columns where the color is present on the color_grid
    
    # Sample pixels from the center of each grid cell
    for i in range(size):
        for j in range(size):
            # Calculate pixel coordinates proportionally, sampling from cell center
            x = int((j + 0.5) / size * w)
            y = int((i + 0.5) / size * h)
            
            # Get BGR pixel value
            pixel = image[y, x]
            
            # Convert BGR to RGB for consistent color representation
            pixel_rgb = pixel[::-1]
            
            # Check if color exists in set, add if not
            if not any(np.array_equal(pixel_rgb, existing) for existing in color_set):
                color_set.append(pixel_rgb)
            
            # Get the index of the current color
            color_index = next(idx for idx, existing in enumerate(color_set) 
                               if np.array_equal(pixel_rgb, existing))
            
            # Assign color index to grid
            color_grid[i, j] = color_index
            
            # Populate residesDict
            if color_index not in residesDict:
                residesDict[color_index] = []
            residesDict[color_index].append((i, j))
            
            # Populate color_rows
            if color_index not in color_rows:
                color_rows[color_index] = []
            if i not in color_rows[color_index]:
                color_rows[color_index].append(i)
            
            # Populate color_columns
            if color_index not in color_columns:
                color_columns[color_index] = []
            if j not in color_columns[color_index]:
                color_columns[color_index].append(j)
    
    # Display the grid
    print("Color Grid:")
    for row in color_grid:
        print(" ".join(map(str, row)))
    
    # Print out the dictionaries
    print("\nResides Dictionary:")
    for color_idx, positions in residesDict.items():
        print(f"Color {color_idx}: {positions}")
    
    print("\nColor Rows:")
    for color_idx, rows in color_rows.items():
        print(f"Color {color_idx}: {rows}")
    
    print("\nColor Columns:")
    for color_idx, columns in color_columns.items():
        print(f"Color {color_idx}: {columns}")
    
    # Optional: Visualize color palette
    plt.figure(figsize=(10, 2))
    plt.title("Color Palette")
    for i, color in enumerate(color_set):
        plt.bar(i, 1, color=color/255, width=1)
    plt.show()
    
    return color_grid, color_set, residesDict, color_rows, color_columns

# Main execution
grid_image, w, h = crop_image("queen1.png")
cv2_imshow(grid_image)
size = count_columns(grid_image)
color_grid, color_set, residesDict, color_rows, color_columns = color_map(grid_image, size, w, h)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow

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

def count_columns(image):
    # Convert the line to grayscale if it's a color image
    if len(image.shape) == 3:
        line = cv2.cvtColor(image[10:11, :], cv2.COLOR_BGR2GRAY)
    else:
        line = image[10:11, :]

    # Threshold the image to create a binary representation
    _, binary_line = cv2.threshold(line, 100, 255, cv2.THRESH_BINARY)
    # cv2.threshold( src, thresh, maxval, type[, dst] ) → retval, dst
    # src: input array (single-channel, 8-bit or 32-bit floating point).
    # thresh: threshold value.
    # maxval: maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    # type: thresholding type (THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV).
    # dst: output array of the same size and type as src.

    
    # Flatten the binary line to 1D array
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


# Main execution
grid_image, w, h = crop_image("queen1.png")
cv2_imshow(grid_image)
size = count_columns(grid_image)

# please fix count_columns function so that rather than checking for exactly black, it looks for pixels that are close to black

def color_map(image, size, w, h):
    # Create a 2D array of 0's with size x size
    color_grid = np.zeros((size, size), dtype=int)
    color_set = [0]      
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
    print()

    # Display residesDict
    print("Resides Dict:")
    for color_idx, positions in residesDict.items():
        print(f"Color {color_idx}: {positions}")
    print()

    # Display color_rows
    print("Color Rows:")
    for color_idx, rows in color_rows.items():
        print(f"Color {color_idx}: {rows}")
    print()

    # Display color_columns
    print("Color Columns:")
    for color_idx, columns in color_columns.items():
        print(f"Color {color_idx}: {columns}")
    print()
    return color_grid, color_set, residesDict, color_rows, color_columns


color_grid, color_set, residesDict, color_rows, color_columns = color_map(grid_image, size, w, h)

def reduce(residesDict, color_rows, color_columns):
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    elimination_occurred = True
    solved_colors = dict()
    while elimination_occurred:
        elimination_occurred = False

        # For each color
        for color_num in range(1, size+1):
            if color_num in solved_colors:
                continue

            # Case 0: Color resides in only 1 cell
            if len(residesDict[color_num]) == 1:
                # Get the row and column of the color
                i, j = residesDict[color_num][0]
                solved_colors[color_num] = (i, j)
            
                # Eliminate row
                for row in range(size):
                    if color_grid[row, j] > 0:
                        color_grid[row, j] = 0
                        elimination_occurred = True

                # Eliminate column
                for column in range(size):
                    if color_grid[i, column] > 0:
                        color_grid[i, column] = 0
                        elimination_occurred = True
                
                # Eliminate adjacent corners if they exist
                for di, dj in directions:
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < size and 0 <= nj < size and color_grid[ni, nj] > 0:
                        color_grid[ni, nj] = 0
                        elimination_occurred = True
                    
            # Case 1: Color resides in 1 row
            if len(color_rows[color_num]) == 1:
                
                row = color_rows[color_num][0]
                for j in range(size):
                    if color_grid[row, j] != color_num and color_grid[row, j] > 0:
                        color_grid[row, j] = 0
                        elimination_occurred = True
            
            # Case 2: Color resides in 1 column
            if len(color_columns[color_num]) == 1:
                column = color_columns[color_num][0]
                for i in range(size):
                    if color_grid[i, column] != color_num and color_grid[i, column] > 0:
                        color_grid[i, column] = 0
                        elimination_occurred = True
            
        # Case 3: Multiple colors confined to the same rows or columns
        row_combos = dict()
        for color_num in range(1, size+1):
            combo = color_rows[color_num]
            frozen_combo = frozenset(combo)
            if frozen_combo in row_combos:
                row_combos[frozen_combo].append(color_num)
            else:
                row_combos[frozen_combo] = [color_num]

        for combo in row_combos:
            color_count = len(row_combos[combo])
            if len(combo) == color_count: # if the number of colors is equal to the number of rows
                for row in combo:
                    for column in range(size):
                        if color_grid[row, column] not in row_combos[combo] and color_grid[row, column] > 0:
                            color_grid[row, column] = 0
                            elimination_occurred = True
                        
        column_combos = dict()
        for color_num in range(1, size+1):
            combo = color_columns[color_num]
            frozen_combo = frozenset(combo)
            if frozen_combo in column_combos:
                column_combos[frozen_combo].append(color_num)
            else:
                column_combos[frozen_combo] = [color_num]

        for combo in column_combos:
            color_count = len(column_combos[combo])
            if len(combo) == color_count:
                for column in combo:
                    for row in range(size):
                        if color_grid[row, column] not in column_combos[combo] and color_grid[row, column] > 0:
                            color_grid[row, column] = 0
                            elimination_occurred = True



        if elimination_occurred:
            # must recompute residesDict, color_rows, color_columns
            residesDict = dict()
            color_rows = dict()
            color_columns = dict()

            for i in range(size):
                for j in range(size):
                    color_index = color_grid[i, j]
                    if color_index > 0:
                        if color_index not in residesDict:
                            residesDict[color_index] = []
                        residesDict[color_index].append((i, j))
                        
                        if color_index not in color_rows:
                            color_rows[color_index] = []
                        if i not in color_rows[color_index]:
                            color_rows[color_index].append(i)
                        
                        if color_index not in color_columns:
                            color_columns[color_index] = []
                        if j not in color_columns[color_index]:
                            color_columns[color_index].append(j)
    return color_grid

reduced_grid = reduce(residesDict, color_rows, color_columns)
print("Reduced Grid:")
for row in reduced_grid:
    print(" ".join(map(str, row)))
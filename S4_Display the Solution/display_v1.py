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
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle variations in line darkness
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Perform morphological operations to enhance line structures
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    vertical_lines = cv2.morphologyEx(morph, cv2.MORPH_OPEN, vertical_kernel)

    # Find contours of vertical lines
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count vertical lines
    column_count = len(contours) - 1  # Subtract 1 to account for the final edge

    print(f"Number of columns: {column_count}")
    
    return column_count


# Main execution
grid_image, w, h = crop_image("queen1.png")
cv2_imshow(grid_image)
size = count_columns(grid_image)


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
    return color_grid, residesDict, color_rows, color_columns


color_grid, residesDict, color_rows, color_columns = color_map(grid_image, size, w, h)


def reduce(residesDict, color_rows, color_columns):
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    elimination_occurred = True
    solved_colors = dict()
    
    while elimination_occurred:
        elimination_occurred = False

        # Create copies of dictionaries to avoid runtime modification issues
        current_residesDict = residesDict.copy()
        current_color_rows = color_rows.copy()
        current_color_columns = color_columns.copy()

        # For each color
        for color_num in range(1, size+1):
            if color_num in solved_colors:
                continue

            # Case 0: Color resides in only 1 cell
            if color_num in current_residesDict and len(current_residesDict[color_num]) == 1:
                # Get the row and column of the color
                i, j = current_residesDict[color_num][0]
                solved_colors[color_num] = (i, j)
            
                # Eliminate row
                for row in range(size):
                    if color_grid[row, j] > 0 and row != i:
                        color_grid[row, j] = 0
                        elimination_occurred = True

                # Eliminate column
                for column in range(size):
                    if color_grid[i, column] > 0 and column != j:
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
            if color_num in current_color_rows and len(current_color_rows[color_num]) == 1:
                row = current_color_rows[color_num][0]
                for j in range(size):
                    if color_grid[row, j] != color_num and color_grid[row, j] > 0:
                        color_grid[row, j] = 0
                        elimination_occurred = True
            
            # Case 2: Color resides in 1 column
            if color_num in current_color_columns and len(current_color_columns[color_num]) == 1:
                column = current_color_columns[color_num][0]
                for i in range(size):
                    if color_grid[i, column] != color_num and color_grid[i, column] > 0:
                        color_grid[i, column] = 0
                        elimination_occurred = True
            
        # Case 3: Multiple colors confined to the same rows or columns
        row_combos = dict()
        for color_num in range(1, size+1):
            if color_num in current_color_rows:
                combo = current_color_rows[color_num]
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
            if color_num in current_color_columns:
                combo = current_color_columns[color_num]
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
            # Recompute residesDict, color_rows, color_columns
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
    
    # Display the grid
    print("Reduced Color Grid:")
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

    return color_grid, residesDict, color_rows, color_columns


reduced_grid, residesDict, color_rows, color_columns = reduce(residesDict, color_rows, color_columns)


# Rules for solution
# Each row, column, and colored region must contain exactly one Crown symbol (Queen).
# Crown symbols cannot be placed in adjacent cells, including diagonally.

def is_valid_placement(grid, row, col, size):
    # Check row and column constraints
    for i in range(size):
        if grid[row][i] == 1 or grid[i][col] == 1:
            return False
    
    # Check diagonal and adjacent constraints
    # directions = [
    #     (-1, -1), (-1, 0), (-1, 1),
    #     (0, -1), (0, 1),
    #     (1, -1), (1, 0), (1, 1)
    # ]
    directions = [
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 1:
            return False
    
    return True


def solve(reduced_grid, residesDict, color_rows, color_columns):
    size = reduced_grid.shape[0]
    queens_grid = np.zeros_like(reduced_grid, dtype=int)
    color_queens = {}  # Track queens for each color
    
    def backtrack(current_color):
        # Base case: All colors have been processed
        if current_color > max(residesDict.keys()):
            return True
        
        # Skip colors that have already been processed
        if current_color not in residesDict:
            return backtrack(current_color + 1)
        
        # Try placing a queen in each possible cell for this color
        for pos in residesDict[current_color]:
            row, col = pos
            
            # Check if a queen can be placed
            if is_valid_placement(queens_grid, row, col, size):
                # Place the queen
                queens_grid[row][col] = 1
                color_queens[current_color] = (row, col)
                
                # Check region constraints
                if all_color_queens_valid(color_queens, reduced_grid):
                    # Recursively try to place queens for the next color
                    if backtrack(current_color + 1):
                        return True
                
                # Backtrack
                queens_grid[row][col] = 0
                color_queens.pop(current_color, None)
        
        return False
    

    def all_color_queens_valid(placed_queens, grid):
        # Check that each color region has exactly one queen
        color_counts = {}
        for color, (row, col) in placed_queens.items():
            # Count how many cells of this color exist
            color_cells = len([pos for pos in residesDict[color]])
            
            # Check that we have a queen in a cell of this color
            color_count = sum(1 for r in range(size) for c in range(size) 
                               if grid[r][c] == color and queens_grid[r][c] == 1)
            
            if color_count > 1:
                return False
        
        return True
    
    # Attempt to solve
    if backtrack(1):  # Start with color 1
        print("Solution Found!")
        print("Queen Placement Grid:")
        print(queens_grid)
        
        print("\nQueen Locations by Color:")
        for color, (row, col) in color_queens.items():
            print(f"Color {color}: Row {row}, Column {col}")
        
        return color_queens
    else:
        print("No solution exists.")
        return None


# Solve the puzzle
color_solution = solve(reduced_grid, residesDict, color_rows, color_columns)

# Display the solution
def display_solution(color_solution, grid_image, w, h, size):
    # Case 1: No solution
    if color_solution is None:
        print("No solution to display.")
        return
    
    # Case 2: Solution Found
    # Create a copy of the original image
    solution_image = grid_image.copy()

    # Mark all queen positions on the image with a black circle
    for color, (row, col) in color_solution.items():
        # Calculate the center of the cell
        x = int((col + 0.5) / size * w)
        y = int((row + 0.5) / size * h)
        
        # Draw a black circle at the center of the cell
        cv2.circle(solution_image, (x, y), 10, (0, 0, 0), -1)

    # Display the solution
    return cv2_imshow(solution_image)

# Display the solution
solution_image = display_solution(color_solution, grid_image, w, h, size)
cv2_imshow(solution_image)
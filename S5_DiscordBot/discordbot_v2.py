import discord
import numpy as np
import cv2
import numpy as np
import io
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the token from environment variables
TOKEN = os.getenv('DISCORD_TOKEN')


# Replace the google.colab.patches import with PIL for image display
def cv2_imshow(image):
    # Convert OpenCV image to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def crop_image(image_array):
    # Check if the image was loaded successfully
    if image_array is None:
        print("Error: Image not found.")
        return False
    else:
        # Get image dimensions
        image_height, image_width, _ = image_array.shape
        print(f"Image Width: {image_width} pixels")
        print(f"Image Height: {image_height} pixels")

        # Dynamically adjust cropping percentages based on image size
        # These percentages worked for the computer image (400x866)
        # We'll create a scaling factor to adapt to different image sizes
        
        # Calculate scaling factor based on the original image width
        base_width = 400  # Width of the original computer image
        width_scale = image_width / base_width

        # Adjust crop percentages with scaling
        top_pct = (150 * width_scale) / image_height
        bottom_pct = (518 * width_scale) / image_height
        left_pct = (15 * width_scale) / image_width
        right_pct = (385 * width_scale) / image_width

        # Convert percentages to pixel values dynamically
        top = int(image_height * top_pct)
        bottom = int(image_height * bottom_pct)
        left = int(image_width * left_pct)
        right = int(image_width * right_pct)

        # final width and height
        width = right - left
        height = bottom - top

        # Perform the cropping
        cropped_image = image_array[top:bottom, left:right]

        # Optional: Resize the cropped image to a standard size if needed
        standard_width = 400
        aspect_ratio = width / height
        standard_height = int(standard_width / aspect_ratio)
        cropped_image = cv2.resize(cropped_image, (standard_width, standard_height))

        return cropped_image, standard_width, standard_height

# Count the number of columns in the grid
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

# Create a color map based on the image
def color_map(image, size, w, h):
    # Create a 2D array of 0's with size x size
    color_grid = np.zeros((size, size), dtype=int)
    color_set = [0]      
    residesDict = dict()  # key = index of color_set, value = list of row and column pairs on the color_grid
    color_rows = dict()   # key = index of color_set, value = list of rows where the color is present on the color_grid
    color_columns = dict()   # key = index of color_set, value = list of columns where the color is present on the color_grid
    
    # Create a copy of the image for visualization
    sample_image = image.copy()
    
    # Sample pixels from the center of each grid cell
    for i in range(size):
        for j in range(size):
            # Calculate pixel coordinates proportionally, sampling from cell center
            # Use the same scaling logic as in crop_image
            base_width = 400  # Consistent with crop_image
            width_scale = w / base_width
            
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
            
            # Visualize sampled pixel with a white circle
            cv2.circle(sample_image, (x, y), 5, (255, 255, 255), -1)
    
    # Optional: Save the sample visualization image
    cv2.imwrite('color_map_samples.png', sample_image)
    
    return color_grid, residesDict, color_rows, color_columns, sample_image

# Reduce the grid based on the color constraints
def reduce(residesDict, color_rows, color_columns, size, color_grid):
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

    return color_grid, residesDict, color_rows, color_columns

# Rules for Placement
# Each row, column, and colored region must contain exactly one Crown symbol (Queen).
# Crown symbols cannot be placed in adjacent cells, including diagonally.
def is_valid_placement(grid, row, col, size):
    # Check row and column constraints
    for i in range(size):
        if grid[row][i] == 1 or grid[i][col] == 1:
            return False
    
    # Check diagonal and adjacent constraints
    directions = [
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 1:
            return False
    
    return True

# Solve the puzzle using backtracking
def solve(reduced_grid, residesDict):
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
        return color_queens
    else:
        return None

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
        # Crop the image
        grid_image, w, h = crop_image(image_array)

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

# Discord Bot Integration
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

# Update the Discord bot message handling to accommodate the new return
@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the message has an attachment
    if message.attachments:
        for attachment in message.attachments:
            # Check if the attachment is an image
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']):
                try:
                    # Download the image
                    image_bytes = await attachment.read()
                    
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Solve the puzzle
                    solution_image, sample_image = solve_queens_puzzle(image_array)
                    
                    if solution_image is not None:
                        # Convert solution image to bytes for Discord
                        solution_bytes = cv2_imshow(solution_image)
                        sample_bytes = cv2_imshow(sample_image)
                        
                        # Send the solution and sample images
                        await message.channel.send(
                            "Here's the solution to the Queens Puzzle!", 
                            file=discord.File(solution_bytes, filename='solution.png')
                        )
                        await message.channel.send(
                            "Here are the sampled pixels for color mapping:", 
                            file=discord.File(sample_bytes, filename='color_map_samples.png')
                        )
                    else:
                        await message.channel.send("Sorry, I couldn't solve the puzzle. Make sure the image is in the correct format.")
                
                except Exception as e:
                    await message.channel.send(f"An error occurred: {e}")

# Use TOKEN when running the bot (from .env file)
client.run(TOKEN)
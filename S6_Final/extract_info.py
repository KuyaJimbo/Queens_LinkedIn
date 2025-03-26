# Import necessary libraries
import cv2
import numpy as np

# Preprocess the image to a standard size
def preprocess_image(image_array):
    # Target dimensions based on the original working image
    target_width = 400

    # Get original image dimensions
    original_height, original_width = image_array.shape[:2]

    # Calculate the scaling factor to fit the target width while maintaining aspect ratio
    scale_factor = target_width / original_width

    # Calculate new dimensions
    new_width = target_width
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

# Crop the image to the desired region
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

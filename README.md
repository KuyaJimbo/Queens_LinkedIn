# LinkedIn Queens Solver Discord Bot

## Project Overview

This Discord bot is an innovative solution for solving the LinkedIn Queens Puzzle, a unique grid-based challenge where queens must be strategically placed according to specific color-based constraints. The bot allows users to upload puzzle images directly to Discord, automatically analyzing and solving the puzzle.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithmic Approach](#algorithmic-approach)
- [Complexity Analysis](#complexity-analysis)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload puzzle images directly to Discord
- Automatically detect grid size and color constraints
- Solve Queens Puzzle using advanced backtracking algorithm
- Provide visual solution with queen placements
- Support for various image formats (PNG, JPG, JPEG, GIF, BMP)

## Installation

### Prerequisites

- Python 3.8+
- Discord Account
- Discord Developer Portal Access

### Dependencies

```bash
pip install discord.py
pip install opencv-python
pip install numpy
pip install python-dotenv
```

### Setup

1. Clone the repository
2. Create a `.env` file with your Discord Bot Token
3. Install required dependencies
4. Run the Discord bot

## Usage

1. Invite the bot to your Discord server
2. Upload a LinkedIn Queens Puzzle image
3. The bot automatically processes and solves the puzzle
4. Receive the solution image with queen placements

## Project Structure

### 1. `backtracking.py`

Implements the core solving algorithm using backtracking.

#### Key Functions:

- `is_valid_placement(grid, row, col, size)`: Validates queen placement
- `solve(reduced_grid, residesDict)`: Main backtracking solver
- `all_color_queens_valid(placed_queens, grid)`: Validates color region constraints

### 2. `discord_bot.py`

Manages Discord bot interactions and image processing workflow.

#### Key Functions:

- `on_message(message)`: Handles incoming Discord messages
- Image download and processing
- Solution generation and sending

### 3. `extract_info.py`

Handles image preprocessing and information extraction.

#### Key Functions:

- `preprocess_image(image_array)`: Standardizes image dimensions
- `crop_image(image_array)`: Extracts puzzle grid
- `count_columns(image)`: Determines grid size
- `color_map(image, size, w, h)`: Creates color-based grid representation

### 4. `reduce_grid.py`

Applies constraint reduction to simplify puzzle solving.

#### Key Functions:

- `reduce(residesDict, color_rows, color_columns, size, color_grid)`: Eliminates impossible placements
- Implements multiple reduction strategies

### 5. `solve_puzzle.py`

Orchestrates the entire solving process.

#### Key Functions:

- `solve_queens_puzzle(image_array)`: Main puzzle-solving pipeline
- `display_solution(color_solution, grid_image, w, h, size)`: Visualizes solution

## Algorithmic Approach

### Puzzle Solving Methodology

1. **Image Preprocessing**

   - Standardize image dimensions
   - Crop to puzzle grid
   - Detect grid size

2. **Color Mapping**

   - Sample pixel colors from grid cells
   - Create color-based grid representation

3. **Constraint Reduction**

   - Apply logical elimination rules
   - Reduce possible queen placements
   - Handle color region constraints

4. **Backtracking Solution**
   - Recursively place queens
   - Validate placement against rules
   - Backtrack when constraints are violated

## Complexity Analysis

### Time Complexity

- **Preprocessing**: O(n²)
- **Color Mapping**: O(n²)
- **Constraint Reduction**: O(n³)
- **Backtracking**: O(n!)

### Space Complexity

- **Grid Representation**: O(n²)
- **Recursive Call Stack**: O(n)
- **Auxiliary Data Structures**: O(n²)

## Technologies Used

- Python
- OpenCV
- NumPy
- Discord.py
- Image Processing Libraries

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a pull request

## License

[Insert Appropriate License]

## Contact

[Your Name/Contact Information]

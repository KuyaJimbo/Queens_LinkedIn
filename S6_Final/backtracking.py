import numpy as np

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

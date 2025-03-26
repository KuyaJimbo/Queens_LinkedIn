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

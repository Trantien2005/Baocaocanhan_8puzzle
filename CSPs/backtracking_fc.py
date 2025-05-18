from Informed_Search.greedy import manhattan_distance

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

def get_children(state):
    """
    Generate all valid child states by moving the blank tile.
    Args:
        state (tuple): Current state (9 numbers, 0 is the blank tile).
    Returns:
        list[tuple]: List of valid child states.
    """
    children = []
    zero_idx = state.index(0)
    row, col = zero_idx // 3, zero_idx % 3
    for move in MOVES:
        new_row, new_col = row + move[0], col + move[1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            new_state = list(state)
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            children.append(tuple(new_state))
    return children

def is_valid_move(state, visited):
    """
    Check if a state is valid (not previously visited).
    Args:
        state (tuple): State to check.
        visited (set): Set of visited states.
    Returns:
        bool: True if the state is valid, False otherwise.
    """
    return state not in visited

def forward_check(state, visited, max_depth, current_depth):
    """
    Perform forward checking to ensure the current path is promising.
    Args:
        state (tuple): Current state.
        visited (set): Set of visited states.
        max_depth (int): Maximum allowed depth.
        current_depth (int): Current recursion depth.
    Returns:
        bool: True if the path is promising, False if it's a dead-end.
    """
    if current_depth >= max_depth:
        return False
    children = get_children(state)
    # Check if any child is unvisited and has a reasonable heuristic
    for child in children:
        if child not in visited:
            # Use Manhattan distance as a heuristic to prioritize promising states
            if manhattan_distance(child) <= manhattan_distance(state) + 2:
                return True
    return False

def backtracking_fc_search(state, path, visited, max_depth, current_depth=0):
    """
    Backtracking search with forward checking and depth limit.
    Args:
        state (tuple): Current state.
        path (list[tuple]): Current path.
        visited (set): Set of visited states.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current recursion depth.
    Returns:
        list[tuple] or None: Path to the goal or None if no solution is found.
    """
    if state == GOAL_STATE:
        return path
    if current_depth >= max_depth:
        return None
    if not forward_check(state, visited, max_depth, current_depth):
        return None
    
    visited.add(state)
    children = sorted(get_children(state), key=manhattan_distance)  # Prioritize by heuristic
    for child in children:
        if is_valid_move(child, visited):
            result = backtracking_fc_search(child, path + [child], visited, max_depth, current_depth + 1)
            if result:
                return result
    visited.remove(state)  # Backtrack by removing the state
    return None

def backtracking_fc(initial_state):
    """
    Backtracking with forward checking to solve the 8-puzzle.
    Args:
        initial_state (tuple): Initial state of the puzzle.
    Returns:
        list[tuple] or None: Path from initial state to goal, or None if no solution.
    """
    visited = set()
    max_depth = 50  # Reasonable depth limit for 8-puzzle
    return backtracking_fc_search(initial_state, [initial_state], visited, max_depth)
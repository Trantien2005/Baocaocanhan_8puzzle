from Informed_Search.greedy import manhattan_distance

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def get_children(state):
    """
    Tạo tất cả trạng thái con từ trạng thái hiện tại bằng cách di chuyển ô trống.
    Args:
        state (tuple): Trạng thái hiện tại (9 số, 0 là ô trống).
    Returns:
        list[tuple]: Danh sách các trạng thái con hợp lệ.
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

def search(path, g, bound, visited):
    """
    Tìm kiếm trong không gian với ngưỡng f = g + h.
    Args:
        path (list[tuple]): Đường đi hiện tại.
        g (int): Chi phí đường đi từ trạng thái ban đầu.
        bound (int): Ngưỡng f tối đa.
        visited (set): Tập hợp các trạng thái đã thăm.
    Returns:
        tuple: (ngưỡng mới, đường đi) hoặc (None, đường đi) nếu tìm thấy mục tiêu.
    """
    state = path[-1]
    f = g + manhattan_distance(state)
    if f > bound:
        return f, None
    if state == GOAL_STATE:
        return None, path
    min_bound = float('inf')
    for child in get_children(state):
        if child not in visited:
            visited.add(child)
            path.append(child)
            next_bound, result = search(path, g + 1, bound, visited)
            if result:
                return None, result
            path.pop()
            visited.remove(child)
            min_bound = min(min_bound, next_bound)
    return min_bound, None

def idastar(initial_state):
    """
    Tìm kiếm A* sâu dần với heuristic khoảng cách Manhattan.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
    Returns:
        list[tuple] or None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    bound = manhattan_distance(initial_state)
    path = [initial_state]
    visited = {initial_state}
    while True:
        next_bound, result = search(path, 0, bound, visited)
        if result:
            return result
        if next_bound == float('inf'):
            return None
        bound = next_bound
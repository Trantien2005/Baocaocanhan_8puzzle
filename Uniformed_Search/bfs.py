from collections import deque

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

def bfs(initial_state):
    """
    Tìm kiếm theo chiều rộng để tìm đường ngắn nhất từ trạng thái ban đầu đến mục tiêu.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
    Returns:
        list[tuple] or None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    queue = deque([(initial_state, [initial_state])])  # (trạng thái, đường đi)
    visited = {initial_state}  # Tập hợp các trạng thái đã thăm
    while queue:
        state, path = queue.popleft()  # Lấy trạng thái đầu tiên trong hàng đợi
        if state == GOAL_STATE:
            return path
        for child in get_children(state):
            if child not in visited:
                visited.add(child)
                queue.append((child, path + [child]))
    return None
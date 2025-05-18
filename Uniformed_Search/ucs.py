from heapq import heappush, heappop

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

def ucs(initial_state):
    """
    Tìm kiếm chi phí đồng nhất để tìm đường ngắn nhất từ trạng thái ban đầu đến mục tiêu.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
    Returns:
        list[tuple] or None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    queue = [(0, initial_state, [initial_state])]  # (chi phí, trạng thái, đường đi)
    visited = {initial_state}
    while queue:
        cost, state, path = heappop(queue)
        if state == GOAL_STATE:
            return path
        for child in get_children(state):
            if child not in visited:
                visited.add(child)
                heappush(queue, (cost + 1, child, path + [child]))
    return None
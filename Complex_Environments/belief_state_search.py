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

def belief_state_search(initial_state):
    """
    Tìm kiếm trạng thái niềm tin, giả định không có thông tin cảm biến.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
    Returns:
        list[tuple] or None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    def search(state, path, visited):
        if state == GOAL_STATE:
            return path
        if state in visited:
            return None
        visited.add(state)
        children = sorted(get_children(state), key=manhattan_distance)
        for child in children:
            result = search(child, path + [child], visited)
            if result:
                return result
        visited.remove(state)
        return None
    return search(initial_state, [initial_state], set())
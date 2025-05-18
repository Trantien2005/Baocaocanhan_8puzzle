import random
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

def stochastic_hill_climbing(initial_state):
    """
    Leo đồi ngẫu nhiên, chọn ngẫu nhiên trạng thái con tốt hơn hiện tại.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
    Returns:
        list[tuple]: Đường đi từ trạng thái ban đầu đến trạng thái cuối cùng (có thể không phải mục tiêu).
    """
    current = initial_state
    path = [current]
    while True:
        if current == GOAL_STATE:
            return path
        children = get_children(current)
        current_h = manhattan_distance(current)
        better_children = [child for child in children if manhattan_distance(child) < current_h]
        if not better_children:
            return path  # Không tìm thấy trạng thái con tốt hơn
        current = random.choice(better_children)
        path.append(current)
import random
import numpy as np
from Informed_Search.greedy import manhattan_distance

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, xuống, trái, phải
ACTIONS = [0, 1, 2, 3]  # Chỉ số của MOVES

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

def get_valid_actions(state):
    """
    Lấy danh sách hành động hợp lệ từ trạng thái hiện tại.
    Args:
        state (tuple): Trạng thái hiện tại.
    Returns:
        list[int]: Danh sách chỉ số hành động hợp lệ.
    """
    zero_idx = state.index(0)
    row, col = zero_idx // 3, zero_idx % 3
    valid = []
    for i, move in enumerate(MOVES):
        new_row, new_col = row + move[0], col + move[1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            valid.append(i)
    return valid

def apply_action(state, action):
    """
    Áp dụng hành động vào trạng thái hiện tại.
    Args:
        state (tuple): Trạng thái hiện tại.
        action (int): Chỉ số hành động (0: lên, 1: xuống, 2: trái, 3: phải).
    Returns:
        tuple or None: Trạng thái mới hoặc None nếu hành động không hợp lệ.
    """
    zero_idx = state.index(0)
    row, col = zero_idx // 3, zero_idx % 3
    new_row, new_col = row + MOVES[action][0], col + MOVES[action][1]
    if 0 <= new_row < 3 and 0 <= new_col < 3:
        new_idx = new_row * 3 + new_col
        new_state = list(state)
        new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
        return tuple(new_state)
    return None

def q_learning(initial_state, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q-Learning để học chính sách di chuyển từ trạng thái ban đầu đến mục tiêu.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
        episodes (int): Số tập học.
        alpha (float): Tỷ lệ học.
        gamma (float): Hệ số giảm giá.
        epsilon (float): Xác suất chọn hành động ngẫu nhiên.
    Returns:
        list[tuple] or None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    # Khởi tạo bảng Q
    q_table = {}
    
    def get_q_value(state, action):
        """Lấy giá trị Q hoặc khởi tạo nếu chưa có."""
        if state not in q_table:
            q_table[state] = {a: 0.0 for a in ACTIONS}
        return q_table[state][action]
    
    def update_q_value(state, action, reward, next_state):
        """Cập nhật giá trị Q dựa trên công thức Q-Learning."""
        current_q = get_q_value(state, action)
        next_max_q = max(get_q_value(next_state, a) for a in get_valid_actions(next_state)) if next_state else 0
        q_table[state][action] = current_q + alpha * (reward + gamma * next_max_q - current_q)
    
    # Huấn luyện
    for _ in range(episodes):
        state = initial_state
        visited = set()
        while state != GOAL_STATE and state not in visited:
            visited.add(state)
            valid_actions = get_valid_actions(state)
            
            # Epsilon-greedy: chọn hành động ngẫu nhiên hoặc tốt nhất
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = max(valid_actions, key=lambda a: get_q_value(state, a))
            
            # Thực hiện hành động
            next_state = apply_action(state, action)
            if next_state is None:
                continue
            
            # Tính phần thưởng: -1 cho mỗi bước, 100 nếu đạt mục tiêu
            reward = 100 if next_state == GOAL_STATE else -1
            
            # Cập nhật bảng Q
            update_q_value(state, action, reward, next_state)
            state = next_state
    
    # Tạo đường đi từ bảng Q
    state = initial_state
    path = [state]
    visited = set([state])
    max_steps = 100  # Giới hạn bước để tránh vòng lặp vô hạn
    while state != GOAL_STATE and len(path) <= max_steps:
        valid_actions = get_valid_actions(state)
        if not valid_actions:
            break
        action = max(valid_actions, key=lambda a: get_q_value(state, a))
        next_state = apply_action(state, action)
        if next_state is None or next_state in visited:
            break
        state = next_state
        path.append(state)
        visited.add(state)
    
    return path if path[-1] == GOAL_STATE else None
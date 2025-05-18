from Informed_Search.greedy import manhattan_distance

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

def get_children(state):
    """
    Tạo tất cả các trạng thái con hợp lệ bằng cách di chuyển ô trống.
    Tham số:
        state (tuple): Trạng thái hiện tại (9 số, 0 là ô trống).
    Kết quả:
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

def and_or_graph_search(initial_state, max_depth=50):
    """
    Tìm kiếm đồ thị AND-OR cho bài toán 8-puzzle, với giới hạn độ sâu để ngăn chặn đệ quy quá mức.
    Giả định môi trường không xác định, sử dụng tìm kiếm đệ quy với sự hướng dẫn của heuristic.
    Tham số:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
        max_depth (int): Số bước tối đa (độ sâu) được phép trong quá trình tìm kiếm.
    Kết quả:
        list[tuple] hoặc None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy giải pháp.
    """
    def search(state, path, visited, depth):
        """
        Hàm tìm kiếm đệ quy với giới hạn độ sâu.
        Tham số:
            state (tuple): Trạng thái hiện tại.
            path (list[tuple]): Đường đi hiện tại.
            visited (set): Tập hợp các trạng thái đã thăm.
            depth (int): Độ sâu đệ quy hiện tại.
        Kết quả:
            list[tuple] hoặc None: Đường đi đến trạng thái mục tiêu hoặc None nếu không tìm thấy giải pháp.
        """
        if state == GOAL_STATE:
            return path
        if depth >= max_depth:  # Stop if depth limit is reached
            return None
        if state in visited:
            return None
        visited.add(state)
        # Sort children by Manhattan distance to prioritize promising states
        children = sorted(get_children(state), key=manhattan_distance)
        for child in children:
            if child not in visited:
                result = search(child, path + [child], visited, depth + 1)
                if result:
                    return result
        visited.remove(state)  # Backtrack by removing the state
        return None

    visited = set()
    return search(initial_state, [initial_state], visited, 0)
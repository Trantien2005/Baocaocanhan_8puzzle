import random
from Informed_Search.greedy import manhattan_distance

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, xuống, trái, phải

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

def apply_moves(state, moves):
    """
    Áp dụng một chuỗi di chuyển vào trạng thái ban đầu.
    Args:
        state (tuple): Trạng thái ban đầu.
        moves (list[int]): Chuỗi các chỉ số di chuyển (0: lên, 1: xuống, 2: trái, 3: phải).
    Returns:
        tuple: Trạng thái cuối cùng sau khi áp dụng di chuyển.
    """
    current = list(state)
    for move_idx in moves:
        zero_idx = current.index(0)
        row, col = zero_idx // 3, zero_idx % 3
        new_row, new_col = row + MOVES[move_idx][0], col + MOVES[move_idx][1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            current[zero_idx], current[new_idx] = current[new_idx], current[zero_idx]
    return tuple(current)

def fitness(moves, initial_state):
    """
    Tính hàm fitness dựa trên khoảng cách Manhattan của trạng thái cuối cùng.
    Args:
        moves (list[int]): Chuỗi di chuyển.
        initial_state (tuple): Trạng thái ban đầu.
    Returns:
        float: Giá trị fitness (càng thấp càng tốt).
    """
    final_state = apply_moves(initial_state, moves)
    return manhattan_distance(final_state)

def generate_individual(length):
    """
    Tạo một cá thể ngẫu nhiên (chuỗi di chuyển).
    Args:
        length (int): Độ dài chuỗi di chuyển.
    Returns:
        list[int]: Chuỗi di chuyển ngẫu nhiên.
    """
    return [random.randint(0, 3) for _ in range(length)]

def crossover(parent1, parent2):
    """
    Lai ghép hai cá thể để tạo cá thể mới.
    Args:
        parent1 (list[int]): Cá thể cha.
        parent2 (list[int]): Cá thể mẹ.
    Returns:
        list[int]: Cá thể con.
    """
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutate(individual, mutation_rate=0.1):
    """
    Đột biến cá thể với xác suất mutation_rate.
    Args:
        individual (list[int]): Cá thể cần đột biến.
        mutation_rate (float): Xác suất đột biến mỗi gen.
    Returns:
        list[int]: Cá thể sau đột biến.
    """
    return [random.randint(0, 3) if random.random() < mutation_rate else gene for gene in individual]

def genetic_algorithm(initial_state, population_size=100, generations=1000, max_moves=50):
    """
    Thuật toán di truyền để tìm đường đi từ trạng thái ban đầu đến mục tiêu.
    Args:
        initial_state (tuple): Trạng thái ban đầu của bài toán.
        population_size (int): Kích thước quần thể.
        generations (int): Số thế hệ tối đa.
        max_moves (int): Độ dài tối đa của chuỗi di chuyển.
    Returns:
        list[tuple] or None: Đường đi từ trạng thái ban đầu đến mục tiêu, hoặc None nếu không tìm thấy.
    """
    # Khởi tạo quần thể
    population = [generate_individual(max_moves) for _ in range(population_size)]
    
    for _ in range(generations):
        # Đánh giá fitness
        population = sorted(population, key=lambda x: fitness(x, initial_state))
        
        # Kiểm tra nếu tìm thấy mục tiêu
        best_individual = population[0]
        best_state = apply_moves(initial_state, best_individual)
        if best_state == GOAL_STATE:
            # Tái tạo đường đi
            path = [initial_state]
            current = list(initial_state)
            for move_idx in best_individual:
                zero_idx = current.index(0)
                row, col = zero_idx // 3, zero_idx % 3
                new_row, new_col = row + MOVES[move_idx][0], col + MOVES[move_idx][1]
                if 0 <= new_row < 3 and 0 <= new_col < 3:
                    new_idx = new_row * 3 + new_col
                    current[zero_idx], current[new_idx] = current[new_idx], current[zero_idx]
                    path.append(tuple(current))
                    if tuple(current) == GOAL_STATE:
                        return path
            return path  # Trả về đường đi tốt nhất dù chưa đạt mục tiêu
        
        # Chọn lọc và tạo thế hệ mới
        new_population = population[:population_size // 10]  # Giữ 10% tốt nhất
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    
    # Trả về đường đi của cá thể tốt nhất
    best_individual = min(population, key=lambda x: fitness(x, initial_state))
    path = [initial_state]
    current = list(initial_state)
    for move_idx in best_individual:
        zero_idx = current.index(0)
        row, col = zero_idx // 3, zero_idx % 3
        new_row, new_col = row + MOVES[move_idx][0], col + MOVES[move_idx][1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_idx = new_row * 3 + new_col
            current[zero_idx], current[new_idx] = current[new_idx], current[zero_idx]
            path.append(tuple(current))
            if tuple(current) == GOAL_STATE:
                return path
    return path
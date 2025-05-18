import time
import tkinter as tk
from tkinter import ttk, messagebox
from Uniformed_Search.bfs import bfs
from Uniformed_Search.dfs import dfs
from Uniformed_Search.ids import ids
from Uniformed_Search.ucs import ucs
from Informed_Search.greedy import greedy_search, manhattan_distance
from Informed_Search.astar import astar
from Informed_Search.idastar import idastar
from Local_Search.simple_hill_climbing import simple_hill_climbing
from Local_Search.steepest_ascent_hill_climbing import steepest_ascent_hill_climbing
from Local_Search.stochastic_hill_climbing import stochastic_hill_climbing
from Local_Search.simulated_annealing import simulated_annealing
from Local_Search.beam_search import beam_search
from Complex_Environments.and_or_graph_search import and_or_graph_search
from Local_Search.genetic import genetic_algorithm
from Complex_Environments.belief_state_search import belief_state_search
from CSPs.backtracking_search import backtracking_search
from Complex_Environments.sensorless_problem import sensorless_search
from CSPs.backtracking_fc import backtracking_fc
from Reinforcement_Learning.q_learning import q_learning
from Reinforcement_Learning.sarsa import sarsa

# Định nghĩa trạng thái ban đầu và mục tiêu
INITIAL_STATE = (2, 6, 5, 0, 8, 7, 4, 3, 1)
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

class PuzzleGUI:
    def __init__(self, root):
        """
        Khởi tạo giao diện người dùng cho bài toán 8-puzzle.
        Args:
            root: Cửa sổ chính của Tkinter.
        """
        self.root = root
        self.root.title("8-puzzle")
        self.root.geometry("800x700")
        self.root.configure(bg="#e0e7ef")

        # Khung chính
        main_frame = tk.Frame(root, bg="#e0e7ef")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Thêm tiêu đề
        title_label = tk.Label(main_frame, text="8-puzzle", bg="#e0e7ef", font=("Helvetica", 25, "bold"), fg="#333333")
        title_label.pack(pady=(0, 10), fill="x")

        # Khung điều khiển
        control_frame = tk.Frame(main_frame, bg="#e0e7ef", bd=2, relief="groove")
        control_frame.pack(pady=5, fill="x")
        tk.Label(control_frame, text="Chọn thuật toán:", bg="#e0e7ef", font=("Helvetica", 12, "bold"), fg="#333333").pack(side="left", padx=5)
        self.algo_var = tk.StringVar(value="BFS")
        algorithms = ["BFS", "UCS", "DFS", "IDS", "Greedy Search", "A*", "IDA*", 
                      "Simple Hill Climbing", "Steepest-Ascent Hill Climbing", 
                      "Stochastic Hill Climbing", "Simulated Annealing", "Beam Search", 
                      "AND-OR Graph Search", "Genetic Algorithm", "Belief State Search", 
                      "Backtracking Search", "Sensorless Search", 
                      "Backtracking with Forward Checking", "Q-Learning", "SARSA"]
        self.algo_menu = ttk.Combobox(control_frame, textvariable=self.algo_var, values=algorithms, width=20, font=("Helvetica", 10))
        self.algo_menu.pack(side="left", padx=5)
        self.solve_button = tk.Button(control_frame, text="Giải bài toán", command=self.solve_puzzle, font=("Helvetica", 10), bg="#4CAF50", fg="white", activebackground="#45a049", bd=0, padx=5, pady=2)
        self.solve_button.pack(side="left", padx=2)
        self.reset_button = tk.Button(control_frame, text="Reset", command=self.reset_puzzle, font=("Helvetica", 10), bg="#f44336", fg="white", activebackground="#e53935", bd=0, padx=5, pady=2)
        self.reset_button.pack(side="left", padx=2)

        # Khung hiển thị lưới
        grid_frame = tk.Frame(main_frame, bg="#e0e7ef")
        grid_frame.pack(pady=5)

        # Lưới trạng thái ban đầu
        initial_frame = tk.LabelFrame(grid_frame, text="Trạng thái ban đầu", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#0288d1", bd=2, relief="flat", labelanchor="n")
        initial_frame.grid(row=0, column=0, padx=10, pady=5)
        self.initial_cells = []
        for i in range(3):
            row = []
            for j in range(3):
                cell = tk.Label(initial_frame, text="", width=4, height=2, font=("Helvetica", 12, "bold"), bg="#e0e0e0", fg="#333333", relief="solid", borderwidth=1)
                cell.grid(row=i, column=j, padx=1, pady=1)
                row.append(cell)
            self.initial_cells.append(row)

        # Lưới trạng thái hiện tại
        current_frame = tk.LabelFrame(grid_frame, text="Quá trình giải", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#388e3c", bd=2, relief="flat", labelanchor="n")
        current_frame.grid(row=0, column=1, padx=10, pady=5)
        self.current_cells = []
        for i in range(3):
            row = []
            for j in range(3):
                cell = tk.Label(current_frame, text="", width=4, height=2, font=("Helvetica", 12, "bold"), bg="#e0e0e0", fg="#333333", relief="solid", borderwidth=1)
                cell.grid(row=i, column=j, padx=1, pady=1)
                row.append(cell)
            self.current_cells.append(row)

        # Lưới trạng thái mục tiêu
        goal_frame = tk.LabelFrame(grid_frame, text="Mục tiêu", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#d32f2f", bd=2, relief="flat", labelanchor="n")
        goal_frame.grid(row=0, column=2, padx=10, pady=5)
        self.goal_cells = []
        for i in range(3):
            row = []
            for j in range(3):
                cell = tk.Label(goal_frame, text="", width=4, height=2, font=("Helvetica", 12, "bold"), bg="#e0e0e0", fg="#333333", relief="solid", borderwidth=1)
                cell.grid(row=i, column=j, padx=1, pady=1)
                row.append(cell)
            self.goal_cells.append(row)

        # Hiển thị trạng thái ban đầu và mục tiêu
        self.display_state(self.initial_cells, INITIAL_STATE)
        self.display_state(self.current_cells, INITIAL_STATE)
        self.display_state(self.goal_cells, GOAL_STATE)

        # Khung hiển thị kết quả
        result_frame = tk.Frame(main_frame, bg="#e0e7ef")
        result_frame.pack(fill="x", pady=5)
        self.move_label = tk.Label(result_frame, text="Bước hiện tại: ", bg="#e0e7ef", font=("Helvetica", 10), fg="#333333")
        self.move_label.pack(anchor="w")
        self.time_label = tk.Label(result_frame, text="Thời gian thực thi: ", bg="#e0e7ef", font=("Helvetica", 10), fg="#333333")
        self.time_label.pack(anchor="w")
        self.steps_label = tk.Label(result_frame, text="Tổng số bước: ", bg="#e0e7ef", font=("Helvetica", 10), fg="#333333")
        self.steps_label.pack(anchor="w")
        self.hn_label = tk.Label(result_frame, text="h(n): ", wraplength=700, bg="#e0e7ef", font=("Helvetica", 10), fg="#333333")
        self.hn_label.pack(anchor="w")

        # Khung hiển thị đường đi
        path_frame = tk.LabelFrame(main_frame, text="Đường đi", font=("Helvetica", 10, "bold"), bg="#ffffff", fg="#333333", bd=2, relief="flat", labelanchor="n")
        path_frame.pack(fill="x", pady=5, padx=10)
        self.canvas = tk.Canvas(path_frame, bg="#ffffff", height=150, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(path_frame, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#ffffff")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="top", fill="x")
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Biến trạng thái
        self.path_cells = []
        self.solution_path = []
        self.current_step = 0
        self.is_animating = False
        self.start_time = 0
        self.step_times = []
        self.hn_values = []
        self.gn_values = []
        self.timer_running = False

    def display_state(self, cells, state):
        """
        Hiển thị trạng thái trên lưới 3x3.
        Args:
            cells: Danh sách các ô Label trong lưới.
            state (tuple): Trạng thái cần hiển thị.
        """
        for i in range(3):
            for j in range(3):
                value = state[i * 3 + j]
                cells[i][j].config(text=str(value) if value != 0 else "", bg=cells[i][j]["bg"])

    def display_path_as_grid(self, path):
        """
        Hiển thị đường đi dưới dạng lưới trong khung cuộn.
        Args:
            path (list[tuple]): Đường đi cần hiển thị.
        """
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.path_cells = []
        if not path:
            return
        for idx, state in enumerate(path):
            state_frame = tk.Frame(self.scrollable_frame, bg="#ffffff", bd=1, relief="solid", highlightbackground="#e0e0e0", highlightthickness=1)
            state_frame.grid(row=0, column=idx, padx=2, pady=2)
            cells = []
            for i in range(3):
                cell_row = []
                for j in range(3):
                    value = state[i * 3 + j]
                    cell = tk.Label(state_frame, text=str(value) if value != 0 else "", width=2, height=1, font=("Helvetica", 10, "bold"), bg="#e0e0e0", fg="#333333", relief="solid", borderwidth=1)
                    cell.grid(row=i, column=j, padx=0, pady=0)
                    cell_row.append(cell)
                cells.append(cell_row)
            self.path_cells.append(cells)
            step_label = tk.Label(state_frame, text=f"Bước {idx}", bg="#ffffff", font=("Helvetica", 8), fg="#333333")
            step_label.grid(row=3, column=0, columnspan=3)
            manhattan_label = tk.Label(state_frame, text=f"Manhattan: {manhattan_distance(state)}", bg="#ffffff", font=("Helvetica", 8), fg="#333333")
            manhattan_label.grid(row=4, column=0, columnspan=3)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_move(self, state1, state2):
        """
        Xác định hướng di chuyển từ state1 sang state2.
        Args:
            state1 (tuple): Trạng thái trước.
            state2 (tuple): Trạng thái sau.
        Returns:
            str: Hướng di chuyển ("UP", "DOWN", "LEFT", "RIGHT") hoặc None.
        """
        zero_idx1 = state1.index(0)
        zero_idx2 = state2.index(0)
        row1, col1 = zero_idx1 // 3, zero_idx1 % 3
        row2, col2 = zero_idx2 // 3, zero_idx2 % 3
        if row1 > row2:
            return "UP"
        elif row1 < row2:
            return "DOWN"
        elif col1 > col2:
            return "LEFT"
        elif col1 < col2:
            return "RIGHT"
        return None

    def update_timer(self):
        """
        Continuously update the time label in seconds while the timer is running.
        """
        if self.timer_running:
            elapsed_time = time.time() - self.start_time  # Time in seconds
            self.time_label.config(text=f"Thời gian thực thi: {elapsed_time:.2f}s")
            self.root.after(10, self.update_timer)  # Update every 10ms

    def animate_solution(self):
        """
        Animate the solving process by updating the "Quá trình giải" grid step by step.
        """
        if self.current_step < len(self.solution_path):
            # Update the current grid to match the current step
            self.display_state(self.current_cells, self.solution_path[self.current_step])
            # Update the move label
            if self.current_step > 0:
                move = self.get_move(self.solution_path[self.current_step - 1], self.solution_path[self.current_step])
                self.move_label.config(text=f"Bước hiện tại: {move}")
            else:
                self.move_label.config(text="Bước hiện tại: Start")
            # Update h(n) and g(n) if applicable
            if self.hn_values and self.algo_var.get() in ["A*", "IDA*", "Simple Hill Climbing", 
                                                         "Steepest-Ascent Hill Climbing", "Stochastic Hill Climbing", 
                                                         "Simulated Annealing", "Beam Search", "AND-OR Graph Search", 
                                                         "Genetic Algorithm", "Belief State Search", "Backtracking Search", 
                                                         "Sensorless Search", "Backtracking with Forward Checking", 
                                                         "Q-Learning", "SARSA"]:
                self.hn_label.config(text=f"h(n): {self.hn_values[self.current_step]}, g(n): {self.gn_values[self.current_step]}")
            elif self.hn_values:
                self.hn_label.config(text=f"h(n): {self.hn_values[self.current_step]}")
            else:
                self.hn_label.config(text="h(n): N/A")
            self.current_step += 1
            if self.current_step < len(self.solution_path):
                self.root.after(500, self.animate_solution)  # Delay of 500ms between steps
            else:
                self.move_label.config(text="Bước hiện tại: Done")
                self.is_animating = False
                self.timer_running = False  # Stop the timer when animation ends
        else:
            self.is_animating = False
            self.timer_running = False

    def solve_puzzle(self):
        """
        Thực thi thuật toán được chọn và lưu đường đi để hiển thị.
        Tính toán h(n) và g(n) cho các thuật toán sử dụng heuristic.
        """
        if self.is_animating:
            return
        self.start_time = time.time()  # Start time at button press
        self.timer_running = True  # Start the timer
        self.update_timer()  # Begin updating the timer
        algo = self.algo_var.get()
        self.step_times = []
        self.hn_values = []
        self.gn_values = []
        if algo == "BFS":
            path = bfs(INITIAL_STATE)
        elif algo == "DFS":
            path = dfs(INITIAL_STATE)
        elif algo == "IDS":
            path = ids(INITIAL_STATE)
        elif algo == "UCS":
            path = ucs(INITIAL_STATE)
        elif algo == "Greedy Search":
            path = greedy_search(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
        elif algo == "A*":
            path = astar(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "IDA*":
            path = idastar(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Simple Hill Climbing":
            path = simple_hill_climbing(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Steepest-Ascent Hill Climbing":
            path = steepest_ascent_hill_climbing(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Stochastic Hill Climbing":
            path = stochastic_hill_climbing(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Simulated Annealing":
            path = simulated_annealing(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Beam Search":
            path = beam_search(INITIAL_STATE, beam_width=2)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "AND-OR Graph Search":
            path = and_or_graph_search(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Genetic Algorithm":
            path = genetic_algorithm(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Belief State Search":
            path = belief_state_search(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Backtracking Search":
            path = backtracking_search(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Sensorless Search":
            path = sensorless_search(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Backtracking with Forward Checking":
            path = backtracking_fc(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "Q-Learning":
            path = q_learning(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        elif algo == "SARSA":
            path = sarsa(INITIAL_STATE)
            if path:
                self.hn_values = [manhattan_distance(state) for state in path]
                self.gn_values = list(range(len(path)))
        else:
            messagebox.showerror("Error", "Vui lòng chọn thuật toán hợp lệ!")
            return
        if path:
            self.solution_path = path
            self.current_step = 0
            self.is_animating = True
            self.step_times.append(self.start_time)
            total_steps = len(path) - 1
            self.steps_label.config(text=f"Tổng số bước: {total_steps}")
            self.display_path_as_grid(path)
            self.move_label.config(text="Bước hiện tại: Start")
            if self.hn_values:
                self.hn_label.config(text=f"h(n): {self.hn_values[0]}")
            else:
                self.hn_label.config(text="h(n): N/A")
            if path[-1] != GOAL_STATE:
                messagebox.showinfo("Kết quả", f"Thuật toán {self.algo_var.get()} không tìm thấy đường đi đến trạng thái mục tiêu sau {total_steps} bước.")
            self.animate_solution()
        else:
            self.display_path_as_grid([])
            self.timer_running = False  # Stop the timer if no path is found
            elapsed_time = time.time() - self.start_time  # Time in seconds
            self.time_label.config(text=f"Thời gian thực thi: {elapsed_time:.2f}s")
            self.steps_label.config(text="Tổng số bước: 0")
            self.move_label.config(text="Bước hiện tại: ")
            self.hn_label.config(text="h(n): ")
            messagebox.showinfo("Kết quả", f"Thuật toán {algo} không tìm thấy đường đi đến trạng thái mục tiêu.")
            self.is_animating = False

    def reset_puzzle(self):
        """Đặt lại giao diện về trạng thái ban đầu, xóa đường đi và thông tin."""
        self.timer_running = False  # Stop the timer on reset
        self.display_state(self.current_cells, INITIAL_STATE)
        self.display_path_as_grid([])
        self.move_label.config(text="Bước hiện tại: ")
        self.time_label.config(text="Thời gian thực thi: ")
        self.steps_label.config(text="Tổng số bước: ")
        self.hn_label.config(text="h(n): ")
        self.solution_path = []
        self.hn_values = []
        self.gn_values = []
        self.step_times = []
        self.current_step = 0
        self.is_animating = False

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()
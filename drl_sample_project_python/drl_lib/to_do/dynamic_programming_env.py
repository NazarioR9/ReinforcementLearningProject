from ..do_not_touch.contracts import MDPEnv
import numpy as np


class LineWorld(MDPEnv):
    def __init__(self, cells_count: int):
        assert (cells_count >= 3)
        self.cells_count = cells_count
        self.S = np.arange(self.cells_count)
        self.A = np.array([0, 1])
        self.R = np.array([-1, 0, 1])
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self.build_env_dynamic()

    def build_env_dynamic(self):
        for i in range(1, self.cells_count - 2):
            self.p[i, 1, i + 1, 1] = 1.0

        for i in range(2, self.cells_count - 1):
            self.p[i, 0, i - 1, 1] = 1.0

        self.p[self.cells_count - 2, 1, self.cells_count - 1, 2] = 1.0
        self.p[1, 0, 0, 0] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        if s in [0, self.cells_count-1]:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]

    def view_state(self, s: int):
        print("State {} is : {}".format(s, self.p[s]))


class GridWorld(MDPEnv):
    def __init__(self, grid_count: int):
        assert (grid_count >= 3)
        self.grid_count = grid_count
        self.max_cells = self.grid_count**2
        self.S = np.arange(self.max_cells)
        self.A = np.array([0, 1, 2, 3])  # 0:up, 1: down, 2: left, 3: right
        self.R = np.array([-1, 0, 1])
        self.p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))
        self.build_env_dynamic()

    def build_env_dynamic(self):
        max_cells = self.max_cells
        # up
        for i in range(self.grid_count, max_cells - 1):
            self.p[i, 0, i - self.grid_count, 1] = 1.0
        # down
        for i in range(0, max_cells - self.grid_count):
            if i == self.grid_count-1:
                continue
            self.p[i, 1, i + self.grid_count, 1] = 1.0
        # left
        for i in range(max_cells - 1):
            if (i % self.grid_count == 0) or (i in [self.grid_count-1, max_cells-1]):
                continue
            self.p[i, 2, i-1, 1] = 1.0
        # right
        for i in range(max_cells - 1):
            if (i+1) % self.grid_count == 0:
                continue
            self.p[i, 3, i+1, 1] = 1.0

        # Go to upper rightmost from neighbours
        self.p[self.grid_count - 2, 3, self.grid_count - 1, 0] = 1.0
        self.p[2*self.grid_count - 1, 0, self.grid_count - 1, 0] = 1.0
        # Go to upper leftmost from neighbours
        self.p[max_cells - self.grid_count - 1, 1, max_cells - 1, 2] = 1.0
        self.p[max_cells - 2, 3, max_cells - 1, 2] = 1.0

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        if s in [0, self.grid_count - 1]:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]

    def view_state(self, s: int):
        print("State {} is : {}".format(s, self.p[s]))

from __future__ import annotations
# nuruomino.py: Projeto de Inteligência Artificial 2024/2025.
# Grupo 35:
# 109686 Miguel Raposo
# 110632 Inês Costa

from search import Problem, depth_first_tree_search
from utils import is_in
import numpy as np
import sys

class Tetromino:
    """Represents a Tetromino with a specific shape, rotation, and reflection."""
    TETROMINO_SHAPES = {
        "L": np.array([
            [1, 0],
            [1, 0],
            [1, 1]
        ]),
        "I": np.array([
            [1],
            [1],
            [1],
            [1]
        ]),
        "T": np.array([
            [1, 1, 1],
            [0, 1, 0]
        ]),
        "S": np.array([
            [0, 1, 1],
            [1, 1, 0]
        ])
    }

    def __init__(self, shape: str, rotation: int = 0, reflected: bool = False):
        self.shape = shape
        self.rotation = rotation
        self.reflected = reflected

    def __repr__(self):
        array = self.get()
        array_str = "\n" + "\n".join("".join('•' if cell else ' ' for cell in row) for row in array)
        return (f'Tetromino(shape:{self.shape}, rotation={self.rotation}º, reflected={self.reflected}){array_str}')

    @staticmethod
    def rotate(tetromino_array, degrees):
        """Rotate a tetromino by the given degrees (must be a multiple of 90)."""
        n = (degrees // 90) % 4
        return Tetromino.normalize(np.rot90(tetromino_array, -n))

    @staticmethod
    def reflect(tetromino_array):
        """Applies a vertical reflection."""
        return Tetromino.normalize(np.fliplr(tetromino_array))

    @staticmethod
    def normalize(tetromino_array):
        """Shift tetromino to top-left corner."""
        rows, cols = np.where(tetromino_array)
        return tetromino_array[rows.min():, cols.min():]

    def get(self):
        """Applies rotation and reflection, returning a set of final coordinates for the tetromino."""
        tetromino_array = Tetromino.TETROMINO_SHAPES[self.shape]
        tetromino_array = Tetromino.rotate(tetromino_array, self.rotation)
        if self.reflected:
            tetromino_array = Tetromino.reflect(tetromino_array)
        return tetromino_array

    @staticmethod
    def get_all_orientations():
        """Returns all unique orientations of Tetromino pieces, normalized."""
        tetrominos_arrays = []
        seen = set()
        for shape in Tetromino.TETROMINO_SHAPES:
            for reflected in (False, True):
                for rotation in range(0, 360, 90):
                    tetromino = Tetromino(shape, rotation, reflected)
                    tetromino_array = tetromino.get()
                    tetromino_id = (shape, tetromino_array.shape, tuple(tetromino_array.flatten()))
                    if tetromino_id not in seen:
                        seen.add(tetromino_id)
                        tetrominos_arrays.append(tetromino)
        return tetrominos_arrays

class Board:
    """Internal representation of a Nuruomino Puzzle board using only numpy arrays."""

    def __init__(self, board: np.matrix, nuruomino: Nuruomino = None):
        self.board = board
        self.nuruomino = nuruomino

    def __repr__(self):
        """Returns a string representation of the board."""
        return "\n".join("\t".join(str(cell) for cell in row) for row in self.board)

    @staticmethod
    def parse_instance():
        """Reads the board from stdin as a numpy matrix and returns a Board instance."""
        return Board(np.loadtxt(sys.stdin, dtype=np.int16).astype(object))

    def set_problem(self, nuruomino: Nuruomino):
        """Sets the Nuruomino problem instance for this board."""
        self.nuruomino = nuruomino

    def get_value(self, row: int, col: int):
        """Returns the value of a given cell on the board."""
        return self.board[row, col]

    def get_cross_adjacent_values(self, row: int, col: int):
        """Returns the values of the cells adjacent to the given cell."""
        adjacent_values = []
        rows, cols = self.board.shape
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                adjacent_values.append(self.get_value(new_row, new_col))
        return adjacent_values

    def get_adjacent_regions(self, row: int, col: int):
        """Returns the regions adjacent to the given cell, including diagonals. Ignores placed L, I, T, S pieces."""
        rows, cols = self.board.shape
        regions = set()
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                cell_value = self.get_value(new_row, new_col)
                if isinstance(cell_value, (int, np.integer)) and cell_value != 0:
                    regions.add(cell_value)
        return regions

    def tetromino_fits_in_region(self, action: Action) -> bool:
        """Check if all coordinates in action.position are within the specified region."""
        for r, c in action.position:
            if not (0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1]):
                return False
            if self.board[r, c] != action.region:
                return False
        return True

    def run_action(self, action: Action) -> Board:
        new_board = self.board.copy()
        for r, c in action.position:
            new_board[r, c] = action.tetromino.shape
        board = Board(new_board, self.nuruomino)
        return board

    def num_shared_edges(self, action: Action):
        """Returns the number of tetromino cells that touch the edge of another region."""
        num_contacts = 0
        for row, col in action.position:
            adjacent_values = self.get_cross_adjacent_values(row, col)
            if any(value != action.region for value in adjacent_values):
                num_contacts += 1
        return num_contacts

    def _tetrominos_fully_connected(self):
        """Check if all tetrominos on the board are connected to each other (doesn't form islands)."""
        union_find = UnionFind()
        tetro_shapes = set(Tetromino.TETROMINO_SHAPES.keys())
        tetrominos = []
        rows, cols = self.board.shape
        for row in range(rows):
            for col in range(cols):
                cell_value = self.get_value(row, col)
                if isinstance(cell_value, str) and is_in(cell_value, tetro_shapes):
                    tetrominos.append((row, col))
                    union_find.make_set((row, col))
        if len(tetrominos) <= 1:
            return True

        tetromino_set = set(tetrominos)
        for row, col in tetrominos:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adj = (row + dr, col + dc)
                if adj in tetromino_set:
                    union_find.union((row, col), adj)

        return union_find.get_components() == 1

    def is_solved(self, regions, filled_regions):
        """Check if all regions are filled, all tetrominos are connected."""
        return len(regions) == len(filled_regions) and self._tetrominos_fully_connected()

class Action:
    """Represents placing a Tetromino at a specific location."""
    def __init__(self, region:int, tetromino:Tetromino, position:list[tuple[int,int]]):
        self.region = region
        self.tetromino = tetromino
        self.position = position

    def __repr__(self):
        pos_str = [(int(r), int(c)) for r, c in self.position]
        return (f'Action(region={self.region}, tetromino_shape={self.tetromino.shape}, position={pos_str})')

    def is_valid(self, board: Board) -> bool:
        return board.tetromino_fits_in_region(self) and board.num_shared_edges(self) > 0

    def is_currently_valid(self, board: Board) -> bool:
        """Check if action doesn't overlap already filled cells, doesn't create 2x2 filled cells, doesn't touch pieces with the same shape."""
        if not board.tetromino_fits_in_region(self):
            return False
        if self._creates_2x2_filled(board):
            return False
        if self._touches_same_shape(board):
            return False
        return True

    def _creates_2x2_filled(self, board: Board) -> bool:
        """Check if placing this tetromino creates any 2x2 filled areas using numpy."""
        temp_board = board.run_action(self)
        arr = temp_board.board
        mask = np.vectorize(lambda x: is_in(x, list(Tetromino.TETROMINO_SHAPES.keys())))(arr)
        filled_2x2 = (mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
        return np.any(filled_2x2)

    def _touches_same_shape(self, board: Board) -> bool:
        """Check if tetromino touches pieces with the same shape."""
        for row, col in self.position:
            adjacent_values = board.get_cross_adjacent_values(row, col)
            if self.tetromino.shape in adjacent_values:
                return True
        return False

class UnionFind:
    """Union-Find data structure for tracking connected components."""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if x not in self.parent:
            return None
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x is None or root_y is None or root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_components(self):
        if not self.parent:
            return 0
        return len(set(self.find(x) for x in self.parent))

class Nuruomino(Problem):
    """Represents the Nuruomino puzzle as a search problem."""
    def __init__(self, board: Board):
        self.og_board = board
        self.current_state = NuruominoState(board)
        self._orientations = self._all_tetrominos_orientations()
        self.adjacency_graph = self._build_adjacency_graph()
        self.all_actions  = self._generate_all_actions()
        super().__init__(self.current_state)
        board.set_problem(self)

    def actions(self, state):
        if self.goal_test(state):
            return []
        region_id = self._select_next_region(state)
        if region_id is None:
            return []
        valid_actions = set()
        for action in self.all_actions.get(region_id, []):
            if action.is_currently_valid(state.board):
                valid_actions.add(action)
        return valid_actions

    def result(self, state, action):
        """Return the state that results from executing the given action."""
        new_filled_regions = state.filled_regions.copy()
        new_filled_regions.add(action.region)
        return NuruominoState(state.board.run_action(action), new_filled_regions)

    def goal_test(self, state):
        """Test if the given state is a goal state."""
        return state.board.is_solved(self.adjacency_graph.keys(), state.filled_regions)

    def _all_tetrominos_orientations(self):
        """Generate all possible orientations of Tetrominos."""
        return Tetromino.get_all_orientations()

    def _build_adjacency_graph(self):
        """Build an adjacency graph for the regions of the board using Board's helper methods."""
        adjacency_graph = {}
        rows, cols = self.og_board.board.shape
        for row in range(rows):
            for col in range(cols):
                region = self.og_board.get_value(row, col)
                if region not in adjacency_graph:
                    adjacency_graph[region] = set()
                for value in self.og_board.get_adjacent_regions(row, col):
                    if value != region:
                        adjacency_graph[region].add(value)
        for region in adjacency_graph:
            adjacency_graph[region] = sorted(adjacency_graph[region])
        return adjacency_graph

    def _select_next_region(self, state):
        regions = set(self.adjacency_graph.keys())
        unfilled = regions - state.filled_regions
        if not unfilled:
            return
        # Find regions with the minimum number of actions
        min_actions = min(len(self.all_actions.get(r)) for r in unfilled)
        min_regions = [r for r in unfilled if len(self.all_actions.get(r)) == min_actions]
        if len(min_regions) == 1:
            return min_regions[0]
        # If tie, pick region with most adjacents
        return max(min_regions, key=lambda r: len(self.adjacency_graph[r]), default=None)

    def _generate_all_actions(self):
        """Generate all valid actions for each region."""
        all_actions = {}
        rows, cols = self.og_board.board.shape
        regions = self.adjacency_graph.keys()
        for region in regions:
            all_actions[region] = []
            for tetromino in self._orientations:
                t_array = tetromino.get()
                t_rows, t_cols = t_array.shape
                for row in range(rows - t_rows + 1):
                    for col in range(cols - t_cols + 1):
                        cells = [(row + r, col + c) for r, c in zip(*np.where(t_array))]
                        action = Action(region, tetromino, cells)
                        if action.is_valid(self.og_board):
                            all_actions[region].append(action)
        return all_actions

class NuruominoState:
    """Represents the state of the Nuruomino puzzle, including the board configuration and a unique state ID."""
    state_id = 0

    def __init__(self, board, filled_regions=None):
        self.board = board
        self.filled_regions = filled_regions or set()
        self.id = NuruominoState.state_id 
        NuruominoState.state_id += 1

    def __lt__(self, other):
        NuruominoState.state_id += 1
        return self.id < other.id

if __name__ == "__main__":
    board = Board.parse_instance()
    problem = Nuruomino(board)
    goal_node = depth_first_tree_search(problem)
    if goal_node:
        print(goal_node.state.board)
    else:
        print("No solution found.")

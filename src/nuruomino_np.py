from __future__ import annotations
# nuruomino.py: Projeto de Inteligência Artificial 2024/2025.
# Grupo 35:
# 109686 Miguel Raposo
# 110632 Inês Costa

import numpy as np
import sys
from search import Problem, depth_first_tree_search

# DONE
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

    def get(self):
        """Applies rotation and reflection, returning a set of final coordinates for the tetromino."""
        tetromino_array = Tetromino.TETROMINO_SHAPES[self.shape]
        tetromino_array = Tetromino.rotate(tetromino_array, self.rotation)
        if self.reflected:
            tetromino_array = Tetromino.reflect(tetromino_array)
        return tetromino_array

    @staticmethod
    def normalize(tetromino_array):
        """Shift tetromino to top-left corner."""
        rows, cols = np.where(tetromino_array)
        return tetromino_array[rows.min():, cols.min():]

    @staticmethod
    def rotate(tetromino_array, degrees):
        """Rotate a tetromino by the given degrees (must be a multiple of 90)."""
        n = (degrees // 90) % 4
        return Tetromino.normalize(np.rot90(tetromino_array, -n))

    @staticmethod
    def reflect(tetromino_array):
        """Applies a vertical reflection."""
        return Tetromino.normalize(np.fliplr(tetromino_array))

    def same_shape(self, other: Tetromino):
        """Check if two Tetrominos have the same shape, ignoring rotation and reflection."""
        return self.shape == other.shape

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
                    key = (shape, tetromino_array.shape, tuple(tetromino_array.flatten()))
                    if key not in seen:
                        seen.add(key)
                        tetrominos_arrays.append(tetromino)
        return tetrominos_arrays

# TODO: EVEYTHING
class Board:
    """Internal representation of a Nuruomino Puzzle board using only numpy arrays."""

    def __init__(self, board: np.matrix):
        self.board = board

    @staticmethod
    def parse_instance():
        """Reads the board from stdin as a numpy matrix and returns a Board instance."""
        return Board(np.loadtxt(sys.stdin, dtype=np.int16))

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
                # Only add regions, not L/I/T/S
                if isinstance(cell_value, (int, np.integer)) and cell_value != 0:
                    regions.add(cell_value)
        return regions

    def get_adjacent_tetrominoes(self, row: int, col: int):
        """Returns the Tetrominoes adjacent to the given cell."""
        adjacent_values = self.get_cross_adjacent_values(row, col)
        tetrominoes = set()
        for value in adjacent_values:
            if value != 0:
                tetrominoes.add(Tetromino(value))
        return tetrominoes

# TODO: is_valid
class Action:
    """Represents placing a Tetromino at a specific location."""
    def __init__(self, region:int, tetromino:Tetromino, position:list[tuple[int,int]]):
        self.region = region
        self.tetromino = tetromino
        self.position = position

    def __repr__(self):
        return (f'Action(region={self.region}, tetromino_shape={self.tetromino.shape}, position={self.position})')

    def is_valid(self, board: Board) -> bool:
        """Check if Tetromino fits in region, doesn't overlap already filled cells, doesn't create 2x2 filled cells, doesn't touch pieces with the same shape and touches the edge of the region."""


# TODO: _generate_all_actions, _select_next_region
class Nuruomino(Problem):
    """Represents the Nuruomino puzzle as a search problem."""
    def __init__(self, board: Board, current_state=None):
        self.og_board = board
        self.current_state = current_state
        self._orientations = self._all_tetrominos_orientations()
       # self.all_actions  = self._generate_all_actions()
        self.adjacency_graph = self._build_adjacency_graph()
        super().__init__(self.og_board)

    def actions(self, state):
        if self.goal_test(state):
            return []
        region_id = self._select_next_region(state)
        valid_actions = set()
        for action in self.all_actions.get(region_id, []):
            if action.is_valid(state.board):
                valid_actions.add(action)
        return valid_actions

    def result(self, state, action):
        """Return the state that results from executing the given action."""
        return NuruominoState(state.board.run_action(action))

    def goal_test(self, state):
        """Test if the given state is a goal state."""
        return state.board.is_solved()

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

    def _select_next_region(self):
        """Select the next region to work on based on the current state."""
        available = [(region, actions) for region, actions in self.all_actions.items() if actions]
        if not available:
            return None
        min_len = min(len(actions) for _, actions in available)
        candidates = [region for region, actions in available if len(actions) == min_len]
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda r: len(self.adjacency_graph.get(r, [])))
    
# DONE
class NuruominoState:
    """Represents the state of the Nuruomino puzzle, including the board configuration and a unique state ID."""
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id 
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """This method is used to break ties in the management of the open list in informed search algorithms."""
        return self.id < other.id

if __name__ == "__main__":
    board = Board.parse_instance()
    problem = Nuruomino(board)
    for key, value in problem.adjacency_graph.items():
        print(f"{key}: {' '.join(map(str, value))}")
    #goal_node = depth_first_tree_search(problem)
    #goal_node.state.board.print_instance()

from __future__ import annotations
# nuruomino.py: Projeto de Inteligência Artificial 2024/2025.
# Grupo 35:
# 109686 Miguel Raposo
# 110632 Inês Costa

import sys
import numpy as np
from search import Problem, depth_first_tree_search

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
        array_str = "\n" + "\n".join(""
            .join('•' if cell else ' ' for cell in row) for row in array)
        return (f'Tetromino(shape:{self.shape}, '
                f'rotation={self.rotation}º, '
                f'reflected={self.reflected}){array_str}')

    @staticmethod
    def normalize(tetromino_array) -> np.matrix:
        """Shift tetromino to top-left corner."""
        rows, cols = np.where(tetromino_array)
        return tetromino_array[rows.min():, cols.min():]

    @staticmethod
    def rotate(tetromino_array, degrees) -> np.matrix:
        """Rotate a tetromino by the given degrees (must be a multiple of 90)."""
        n = (degrees // 90) % 4
        return Tetromino.normalize(np.rot90(tetromino_array, -n))

    @staticmethod
    def reflect(tetromino_array) -> np.matrix:
        """Applies a vertical reflection."""
        return Tetromino.normalize(np.fliplr(tetromino_array))

    def get(self) -> np.matrix:
        """Applies rotation and reflection, returning a set
        of final coordinates for the tetromino."""
        tetromino_array = Tetromino.TETROMINO_SHAPES[self.shape]
        tetromino_array = Tetromino.rotate(tetromino_array, self.rotation)
        if self.reflected:
            tetromino_array = Tetromino.reflect(tetromino_array)
        return tetromino_array

    @staticmethod
    def get_all_orientations() -> np.matrix:
        """Returns all unique orientations of Tetromino pieces, normalized."""
        tetrominos_arrays = []
        seen = set()
        for shape in Tetromino.TETROMINO_SHAPES:
            for reflected in (False, True):
                for rotation in range(0, 360, 90):
                    tetromino = Tetromino(shape, rotation, reflected)
                    tetromino_array = tetromino.get()
                    tetromino_id = (shape, tetromino_array.shape,\
                                    tuple(tetromino_array.flatten()))
                    if tetromino_id not in seen:
                        seen.add(tetromino_id)
                        tetrominos_arrays.append(tetromino)
        return tetrominos_arrays

class Board:
    """Internal representation of a Nuruomino Puzzle board
    using only numpy arrays."""

    def __init__(self, board: np.matrix):
        self.board = board
        self.size = board.shape[0]

    def __repr__(self):
        """Returns a string representation of the board."""
        return "\n".join("\t".join(str(cell)
                for cell in row) for row in self.board)

    @staticmethod
    def parse_instance() -> np.matrix:
        """Reads the board from stdin as a numpy matrix
        and returns a Board instance."""
        return Board(np.loadtxt(sys.stdin, dtype=np.int16).astype(object))

    def copy(self) -> Board:
        """Returns a copy of the board."""
        return Board(self.board.copy())

    def get_value(self, row: int, col: int):
        """Returns the value of a given cell on the board."""
        return self.board[row, col]

    def _get_cross_adjacent_coordinates(self, row: int, col: int):
        """Returns the coordinates of the cells adjacent to the given cell."""
        adjacent_coords = []
        if row > 0:
            adjacent_coords.append((row - 1, col))
        if row < self.size - 1:
            adjacent_coords.append((row + 1, col))
        if col > 0:
            adjacent_coords.append((row, col - 1))
        if col < self.size - 1:
            adjacent_coords.append((row, col + 1))
        return adjacent_coords

    def get_cross_adjacent_values(self, row: int, col: int):
        """Returns the values of the cells adjacent to the given cell."""
        return [self.get_value(adjacent_row, adjacent_col)
            for adjacent_row, adjacent_col
                in self._get_cross_adjacent_coordinates(row, col)]

    def tetromino_fits_in_region(self, action: Action) -> bool:
        """Check if all coordinates filled in action are within the
        specified region."""
        for row, col in action.position:
            if not (0 <= row < self.size and 0 <= col < self.size):
                return False
            if self.board[row, col] != action.region:
                return False
        return True

    def num_shared_edges(self, action: Action):
        """Returns the number of tetromino cells that touch
        the edge of another region."""
        num_contacts = 0
        for row, col in action.position:
            adjacent_values = self.get_cross_adjacent_values(row, col)
            if any(value != action.region for value in adjacent_values):
                num_contacts += 1
        return num_contacts

    def run_action(self, action: Action) -> Board:
        """Returns a new board with the tetromino placed
        at the specified position."""
        for row, col in action.position:
            self.board[row, col] = action.tetromino.shape
        return Board(self.board)

    def get_adjacent_regions(self, region: list[list[int, int]],
        og_board: Board, filled_regions: set) -> set:
        """Returns the ids of the regions adjacent to the given region."""
        adjacent_coordinates = set()
        for row, col in region:
            coordinates = self._get_cross_adjacent_coordinates(row, col)
            for coordinate in coordinates:
                if coordinate not in region:
                    adjacent_coordinates.add(coordinate)
        new_adjacent_regions = set()
        for row, col in adjacent_coordinates:
            value = self.get_value(row, col)
            if isinstance(value, int) and value not in filled_regions:
                new_adjacent_regions.add(value)
            elif value in Tetromino.TETROMINO_SHAPES:
                original_value = og_board.get_value(row, col)
                new_adjacent_regions.add(original_value)
        return new_adjacent_regions

class Action:
    """Represents placing a Tetromino at a specific location."""
    def __init__(self, region:int, tetromino:Tetromino,
        position:list[list[int,int]]):
        self.region = region
        self.tetromino = tetromino
        self.position = position

    def __repr__(self):
        position_str = [(int(row), int(col)) for row, col in self.position]
        return (f'Action(region={self.region}, '
                f'tetromino_shape={self.tetromino.shape}, '
                f'position={position_str})')

    def is_valid(self, board: Board) -> bool:
        """Returns whether a action is valid or not."""
        return board.tetromino_fits_in_region(self) and\
            board.num_shared_edges(self) > 0

    def is_currently_valid(self, board: Board, filled_regions, adjacency_graph) -> bool:
        """Check if action doesn't overlap already filled cells, doesn't
        create 2x2 filled cells and doesn't touch pieces with the same shape."""
        if not board.tetromino_fits_in_region(self):
            return False
        if self._touches_same_tetromino_shape(board):
            return False
        if not self._connected_to_other_tetromino_shape(board, filled_regions, adjacency_graph):
            return False
        if self._creates_filled_2x2_region(board):
            return False
        return True

    def _creates_filled_2x2_region(self, board: Board) -> bool:
        """Check if placing this tetromino would create
        any 2x2 filled areas, without modifying the board."""
        board_array = board.board
        tetromino_shapes = set(Tetromino.TETROMINO_SHAPES.keys())
        positions = set(self.position)
        potencial_2x2_block = set()
        for row, col in self.position:
            for row_offset in (0, -1):
                for col_offset in (0, -1):
                    i, j = row + row_offset, col + col_offset
                    if 0 <= i < board.size - 1 and 0 <= j < board.size - 1:
                        potencial_2x2_block.add((i, j))

        for row, col in potencial_2x2_block:
            if (
                ((board_array[row, col] in tetromino_shapes) or ((row, col) in positions)) and
                ((board_array[row + 1, col] in tetromino_shapes) or ((row + 1, col) in positions)) and
                ((board_array[row, col + 1] in tetromino_shapes) or ((row, col + 1) in positions)) and
                ((board_array[row + 1, col + 1] in tetromino_shapes) or ((row + 1, col + 1) in positions))
            ):
                return True
        return False

    def _touches_same_tetromino_shape(self, board: Board) -> bool:
        """Check if tetromino touches other tetronimo pieces with the same shape."""
        board_array = board.board
        shape = self.tetromino.shape
        for row, col in self.position:
            if row > 0 and board_array[row-1, col] == shape:
                return True
            if row < board.size - 1 and board_array[row+1, col] == shape:
                return True
            if col > 0 and board_array[row, col-1] == shape:
                return True
            if col < board.size - 1 and board_array[row, col+1] == shape:
                return True
        return False

    def _connected_to_other_tetromino_shape(self, board: Board, filled_regions,
        adjacency_graph) -> bool:
        """When all adjacent regions are filled the tetromino placed on the
        region needs to touch a tetromino from one of the adjacent regions."""
        adjacent_regions = adjacency_graph.get(self.region, set())
        if not filled_regions.issuperset(adjacent_regions):
            return True
        tetromino_shapes = Tetromino.TETROMINO_SHAPES
        for row, col in self.position:
            if row > 0 and board.board[row - 1, col] in tetromino_shapes:
                return True
            if row < board.size - 1 and board.board[row + 1, col] in tetromino_shapes:
                return True
            if col > 0 and board.board[row, col - 1] in tetromino_shapes:
                return True
            if col < board.size - 1 and board.board[row, col + 1] in tetromino_shapes:
                return True
        return False

class Nuruomino(Problem):
    """Represents the Nuruomino puzzle as a search problem."""
    def __init__(self, board: Board):
        self.og_board = board
        self._orientations = self._all_tetrominos_orientations()
        self.adjacency_graph = self._build_adjacency_graph()
        self.regions = list(self.adjacency_graph.keys())
        self.current_state = NuruominoState(board,
            self._generate_all_actions(self.adjacency_graph), self.adjacency_graph, set())
        super().__init__(self.current_state)

    def value(self, state):
        pass

    def result(self, state, action):
        """Return the state that results from executing the given action."""
        new_state = NuruominoState(
            board=state.board.copy(),
            actions_graph=state.actions_graph.copy(),
            adjacency_graph={key: value.copy()
                for key, value in state.adjacency_graph.items()},
            filled_regions=state.filled_regions.copy()
        )
        # Update baord.
        new_state.board.run_action(action)
        # Update filled regions.
        new_state.filled_regions.add(action.region)
        # Update adjacency graph.
        new_adjacent_regions = new_state.board.get_adjacent_regions(
            action.position, self.og_board, new_state.filled_regions)
        new_state.adjacency_graph[action.region] = new_adjacent_regions.copy()
        for neighbour in new_adjacent_regions:
            if neighbour in new_state.adjacency_graph:
                new_state.adjacency_graph[neighbour].add(action.region)
        for region, neighbours in new_state.adjacency_graph.items():
            if (region != action.region and action.region in neighbours
                and region not in new_adjacent_regions):
                neighbours.discard(action.region)
        # Update actions graph.
        new_state.actions_graph[action.region] = []
        for region in new_adjacent_regions:
            if region in new_state.actions_graph:
                valid_actions = []
                for new_actions in new_state.actions_graph[region]:
                    if new_actions.is_currently_valid(new_state.board,
                        new_state.filled_regions, self.adjacency_graph):
                        valid_actions.append(new_actions)
                new_state.actions_graph[region] = valid_actions
        return new_state

    def goal_test(self, state):
        """Test if the given state is a goal state."""
        return len(self.regions) == len(state.filled_regions) and\
            self._adjacency_graph_connected(state.adjacency_graph)

    def actions(self, state):
        """Return the actions available in the current state."""
        if not self._adjacency_graph_connected(state.adjacency_graph):
            return []
        return state.actions_graph.get(self._select_next_region(state), [])

    def _all_tetrominos_orientations(self):
        """Generate all possible orientations of Tetrominos."""
        return Tetromino.get_all_orientations()

    def _build_adjacency_graph(self):
        """Build an adjacency graph for the regions of the
        board using Board's helper methods."""
        adjacency_graph = {}
        for row in range(self.og_board.size):
            for col in range(self.og_board.size):
                region = self.og_board.get_value(row, col)
                if region not in adjacency_graph:
                    adjacency_graph[region] = set()
                for value in self.og_board.get_cross_adjacent_values(row, col):
                    if value != region:
                        adjacency_graph[region].add(value)
        return adjacency_graph

    def _generate_all_actions(self, adjacency_graph):
        """Generate all valid actions for each region."""
        all_actions = {}
        regions = adjacency_graph.keys()
        for region in regions:
            all_actions[region] = []
            for tetromino in self._orientations:
                tetromino_array = tetromino.get()
                shape_rows, shape_cols = tetromino_array.shape
                for row in range(self.og_board.size - shape_rows + 1):
                    for col in range(self.og_board.size - shape_cols + 1):
                        cells = [(row + row_offset, col + col_offset)
                            for row_offset, col_offset in zip(*np.where(tetromino_array))]
                        action = Action(region, tetromino, cells)
                        if action.is_valid(self.og_board):
                            all_actions[region].append(action)
        return all_actions

    def _select_next_region(self, state):
        """Selects the next region to explore and test actions."""
        regions = set(state.adjacency_graph.keys())
        regions -= state.filled_regions
        min_actions = min(len(state.actions_graph.get(region, []))
            for region in regions)
        min_regions = [region for region in regions
            if len(state.actions_graph.get(region, [])) == min_actions]

        if len(min_regions) == 0:
            return []
        if len(min_regions) == 1:
            return min_regions[0]
        return max(min_regions, key=lambda region:
            len(state.actions_graph.get(region, [])), default=None)

    def _adjacency_graph_connected(self, adjacency_graph):
        """Return True if the adjacency graph is connected. False otherwise."""
        visited = set()
        queue = [1]
        while queue:
            region = queue.pop(0)
            if region in visited:
                continue
            visited.add(region)
            for neighbour in adjacency_graph[region]:
                if neighbour not in visited:
                    queue.append(neighbour)
        return len(visited) == len(adjacency_graph)

class NuruominoState:
    """Represents the state of the Nuruomino puzzle,
    including the board configuration and a unique state ID."""
    state_id = 0

    def __init__(self, board, actions_graph, adjacency_graph, filled_regions):
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self.board = board
        self.actions_graph = actions_graph
        self.adjacency_graph = adjacency_graph
        self.filled_regions = filled_regions

    def __lt__(self, other):
        NuruominoState.state_id += 1
        return self.id < other.id

def solve_nuruomino():
    """Parses input from stdin, creates a Nuruomino problem and solves it."""
    board = Board.parse_instance()
    problem = Nuruomino(board)
    goal_node = depth_first_tree_search(problem)
    if goal_node:
        print(goal_node.state.board)
    else:
        print("No solution found.")

if __name__ == "__main__":
    solve_nuruomino()
